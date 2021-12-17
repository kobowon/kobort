from typeing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

#[TODO] class 내에서 self를 붙이는 attribute와 self를 붙이지 않는 attribute의 차이는 뭘까
class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: List[Dict[str, str]], #key: str, value: str
        entity_label_list: List[str],
        max_length: int,
        batch_size: int,
        shuffle: bool = False,
        sampler = None
        **kwargs
    ):
        #__getitem__ 을 위한 객체이고, __getitem__ 을 통해 얻어지는 데이터들의 리스트(정확히 말하면 iterator)는 collate_fn Class의 __call__ 함수에 들어갈 첫 번째 인자가 됨
        self.dataset = dataset #List[Union[Dict (여러 값을 담을 수 있음), 특정 값]]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: i for i, label in enumerate(entity_label_list)}
        self.collate_fn = CollateNer(tokenizer, self.label2id, max_length)
        self.loader = DataLoader(
            self, #NerDataset 클래스의 객체이므로 __getitem__ 으로 얻어지는 객체들의 리스트라 생각
            sampler= sampler,
            batch_size = batch_size,
            shuffle = shuffle,
            collate_fn = self.collate_fn,
            **kwargs
        )
    #dataset의 개수 반환
    def __len__(self):
        return len(self.dataset)
    
    #dataset[i]에서 접근가능한 객체들 반환
    def __getitem__(self, i):
        instance = self.dataset[i]
        sentence = instance['sentence']
        label_str = instance['label_str']
        return sentence, label_str


#dataloder에서 부르는 collate 클래스
#입력 : sentences, labels
#출력 : sentences id, labels id
#각각의 데이터를 ids로 변경하는 역할을 함
class CollateNer:
    def __init__(
    self, tokenizer: PreTrainedTokenizer, label2id: Dict[str, int], max_length: int):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    #입력 : __call__에서 받는 examples는 (DataLoader의 첫 번째 인자인 dataset객체)의 __getitem__ method를 통해 가져온 값들의 iterator
    #출력 : __call__에서 반환하는 값은 dataloader의 첫 번째 인자(dataset)로 들어감
    #[TODO] self.tokenizer.batch_encode_plus 메서드를 이용해 리펙토링 예정
    
    def __call__(self, examples):
        
        #이렇게 가져오면 안 되나?
        #sentence_list, label_str_list = examples 
        
        total_input_ids = []
        total_input_attention_mask = []
        total_input_labels_ids = []
        
        for example in examples:
            sentence, label_str = example
            tokens = sentence.split()
            labels = label_str.split()
            
            #texts to ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_labels_ids = [label2id[label] for label in labels]
            
            ##truncate_sequences
            SPECIAL_TOKENS_NUM = 2 #single sentence -> <s>, </s> 두 개
            limit = max_length - SPECIAL_TOKENS_COUNT
            if len(input_ids) > limit:
                input_ids = input_ids[:limit]
                input_labels_ids = input_labels_ids[:limit]
            
            #add special token </s>
            input_ids = input_ids + [tokenizer.sep_token_id]
            input_labels_ids = input_labels_ids + [-100] #-100 : cross entropy loss에서 무시하는 값
            
            #add special token <s> or cls token
            input_ids = [tokenizer.cls_token_id] + input_ids
            input_labels_ids = [-100] + input_labels_ids
            
            input_attention_mask = [1] * len(input_ids)
            padding_num = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_num
            input_attention_mask = input_attention_mask + [0] * padding_num
            input_labels_ids = input_labels_ids + [-100] * padding_num

            assert len(input_ids) == max_length
            assert len(input_attention_mask) == max_length
            assert len(input_labels_ids) == max_length

            if cnt < 3:
                print(f"*** Example {cnt} ***")
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("label_ids: %s" % " ".join(list(map(str,input_labels_ids))))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("attention_mask: %s" % " ".join([str(x) for x in input_attention_mask]))
            
            total_input_ids.append(input_ids)
            total_input_attention_mask.append(input_attention_mask)
            total_input_labels_ids.append(input_labels_ids)
        
        #tensor로 변환
        tensor_ids = torch.tensor(total_input_ids, dtype=torch.long)
        tensor_masks = torch.tensor(total_input_attention_mask, type=torch.long)
        tensor_labels = torch.tensor(total_input_labels_ids, type=torch.long)
        
        return TensorDataset(tensor_ids, tensor_masks, tensor_labels)
