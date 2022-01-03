from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer


#[TODO] class 내에서 self를 붙이는 attribute와 self를 붙이지 않는 attribute의 차이는 뭘까
class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer, #collate 함수에 넘겨줘야함
        dataset: List[Dict[str, str]], #key: str, value: str
        entity_label_list: List[str],
        max_length: int,
        batch_size: int,
        shuffle: bool = False,
        sampler = None,
        **kwargs
    ):
        #__getitem__ 을 위한 객체이고, __getitem__ 을 통해 얻어지는 데이터 batch_size개를 포함한 리스트(정확히 말하면 iterator)는 collate_fn Class의 __call__ 함수에 들어갈 첫 번째 인자가 됨
        self.dataset = dataset #List[Union[Dict (여러 값을 담을 수 있음), 특정 값]]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: i for i, label in enumerate(entity_label_list)}
        self.collate_fn = CollateNer(tokenizer, self.label2id, max_length)
        self.sampler= None
        self.loader = DataLoader(
            dataset=self, #NerDataset객체, __getitem__ 으로 얻어지는 객체들의 리스트라 생각
            batch_size = batch_size,#dataset을 batch_size만큼 가져와서
            shuffle = shuffle,#섞을거면 섞고
            sampler= self.sampler,#섞을거면 섞고 (단 sampler를 쓰려면 shuffle이 False여야함)
            collate_fn = self.collate_fn, #collate_fn로 batch 전달
            # pin_memory=pin_memory,#config → torch.cuda.is_available() 일 때만 사용
            # num_workers=num_workers,#config → torch.cuda.is_available() 일 때만 사용
            **kwargs
        )
    #dataset의 개수 반환
    def __len__(self):
        return len(self.dataset)
    
    #dataset[i]에서 접근가능한 객체들 반환
    def __getitem__(self, i):
        instance = self.dataset[i]
        sentence = instance['tokenized_text_with_unk']
        label_str = instance['label_str']
        return sentence, label_str


#dataloder에서 부르는 collate 클래스
#입력 : sentences, labels
#출력 : sentences id, labels id
#각각의 데이터를 ids로 변경하는 역할을 함
#Dataset의 __getitem__이 혹은 모델의 입력이 (id) 매번 다른 shape의 텐서를 리턴하는 경우, batch_size를 2이상으로 주기 위해서 사용 (shape을 맞춰야하니까)

#__call__ 함수를 통해서 
#dataloader의 dataset을 
#dataloader의 batch_size만큼 가져오고
#collate를 통해 텐서로 변환 (여러 텐서 반환 가능, 각 텐서는 하나로 묶여야 하니 길이가 동일해야함)
class CollateNer:
    def __init__(
    self, tokenizer: PreTrainedTokenizer, label2id: Dict[str, int], max_length: int):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    #입력 : __call__에서 받는 batches는 (DataLoader의 첫 번째 인자인 dataset객체)의 __getitem__ method를 통해 가져온 값들의 iterator
    #출력 : __call__에서 반환하는 값은 dataloader의 첫 번째 인자(dataset)로 들어감
    #[TODO] self.tokenizer.batch_encode_plus 메서드를 이용해 리펙토링 예정
    #loader call될 떄마다 call되는 거 같음
    def __call__(self, batches):
        total_input_ids = []
        total_input_attention_mask = []
        total_input_labels_ids = []
        
        for batch in batches:
            sentence, label_str = batch
            tokens = sentence.split()
            if label_str:
                labels = label_str.split()
            
            #texts to ids
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if label_str:
                input_labels_ids = [self.label2id[label] for label in labels]
            
            ##truncate_sequences
            SPECIAL_TOKENS_NUM = 2 #single sentence -> <s>, </s> 두 개
            limit = self.max_length - SPECIAL_TOKENS_NUM
            if len(input_ids) > limit:
                input_ids = input_ids[:limit]
                if label_str:
                    input_labels_ids = input_labels_ids[:limit]
            
            #add special token </s>
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            if label_str:
                input_labels_ids = input_labels_ids + [-100] #-100 : cross entropy loss에서 무시하는 값
            
            #add special token <s> or cls token
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            if label_str:
                input_labels_ids = [-100] + input_labels_ids
            
            input_attention_mask = [1] * len(input_ids)
            padding_num = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_num
            input_attention_mask = input_attention_mask + [0] * padding_num
            if label_str:
                input_labels_ids = input_labels_ids + [-100] * padding_num

            assert len(input_ids) == self.max_length
            assert len(input_attention_mask) == self.max_length
            if label_str:
                assert len(input_labels_ids) == self.max_length
            
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            total_input_ids.append(input_ids)
            input_attention_mask = torch.tensor(input_attention_mask, dtype=torch.long)
            total_input_attention_mask.append(input_attention_mask)
            if label_str:
                input_labels_ids = torch.tensor(input_labels_ids, dtype=torch.long)
                total_input_labels_ids.append(input_labels_ids)
        
        #tensor를 담은 리스트를 리스트를 담은 tensor로 변환
        tensor_ids = torch.stack(total_input_ids)
        tensor_masks = torch.stack(total_input_attention_mask)
        
        if label_str:
            tensor_labels = torch.stack(total_input_labels_ids)
            return tensor_ids, tensor_masks, tensor_labels
            
        return tensor_ids, tensor_masks
