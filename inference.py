import argparse
import os
import torch
import numpy as np
from utils import set_seed, make_ner_data, return_ner_tagged_sentence_plus
from dataset import NerDataset
from transformers import RobertaForTokenClassification, BertTokenizer


def infer(args):
    #Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                              do_lower_case=False,
                                              unk_token='<unk>',
                                              sep_token='</s>',
                                              pad_token='<pad>',
                                              cls_token='<s>',
                                              mask_token='<mask>',
                                             )
    infer_texts = ["지난 14일 방송된 KBS 2TV '불후의 명곡' 왕중왕전에서는 '쇼쇼쇼, 별들의 귀환' 2편이 꾸며졌다.", "장동민 고소, KBS 쿨 FM 하차…조정치 도희 임시 DJ 맡아 최근 여성 비하성 발언으로 논란이 되며 무한도전 ‘식스맨’을 자진 하차한 장동민이 고소를 당한 것으로 알려졌다." ]
    
    #Build dataloader
    entity_label_list, inference_data = make_ner_data(file_path=None,
                                                      tokenizer=tokenizer,
                                                      texts=infer_texts)
    
    id2label = {str(i): label for i, label in enumerate(entity_label_list)}
    label2id = {label:i for i, label in enumerate(entity_label_list)}
    
    inference_dataset = NerDataset(tokenizer=tokenizer,
                                   dataset=inference_data,
                                   entity_label_list=entity_label_list,
                                   max_length=args.max_length,
                                   batch_size=len(infer_texts),
                                   shuffle=False
                                  )
    dataloader = inference_dataset.loader
    
    #Load model
    model = RobertaForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=len(entity_label_list),
        id2label = id2label,
        label2id = label2id
    )
    model = model.cuda()
    
    #run one time only
    for batch in dataloader:
        batch = tuple(feature.to(device) for feature in batch)
        input_ids, attention_mask = batch
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask
                           )
        
        #get logit
        logits = outputs.logits
        preds_cpu = logits.detach().cpu().numpy() #(batch, seq_len, class)
    
    #추후에 batch 단위로 inference 대비
    preds_cpu = np.argmax(preds_cpu, axis=2) #(batch, seq_len)
    
    batch_num = preds_cpu.shape[0]
    seq_len = preds_cpu.shape[1]
    for batch_idx in range(batch_num):
        print("***********************************************************")
        origin_text = infer_texts[batch_idx]
        #원래 문장 <s> </s> 포함 안 된 문장
        _original_tokens = inference_data[batch_idx]["tokenized_text"].split()
        _tokens_with_unk = inference_data[batch_idx]["tokenized_text_with_unk"].split()
        assert len(_original_tokens) == len(_tokens_with_unk)
        original_tokens = ['<s>'] + _original_tokens + ['</s>']
        tokens_with_unk = ['<s>'] + _tokens_with_unk + ['</s>']
        print(f"\n문장 {batch_idx}의 결과는 아래와 같습니다")
        origin_tokens_labels = []
        #original_tokens의 실질적인 토큰 범위 내에서 (special token 제외)
        for token_idx in range(1,len(original_tokens)-1): #유효한 토큰들만 살펴보기 <pad> 제외
            #예측한 label id를 label로 변환
            pred_label_index = preds_cpu[batch_idx][token_idx]
            pred_label_str = id2label[str(pred_label_index)]
            origin_tokens_labels.append(pred_label_str)
            print(f'{token_idx}번째 토큰 "{original_tokens[token_idx]}→{tokens_with_unk[token_idx]}"의 라벨은 "{pred_label_str}"입니다')
    
        print(f"\n원문장 : {origin_text}")
        print(f"토크나이즈된 <unk> 미포함 문장 : {_original_tokens}")
        print(f"토크나이즈된 <unk> 포함 문장 : {_tokens_with_unk}")
        print(f"토큰에 대한 라벨 리스트 : {origin_tokens_labels}") 
        
        ner_tagged_sentence, ner_tagged_id_list = return_ner_tagged_sentence_plus(origin_text, _original_tokens,origin_tokens_labels)
  
        print(f'태깅 문장 : {ner_tagged_sentence}')
        for item in ner_tagged_id_list:
            start, end, entity = item
            print(f'{entity} : {origin_text[start:end+1]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        default='/data/bowon_ko/TRoBERTa_BASE/220103/finetune/ner/'#210727
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/model/wordpiece_mecab/version_1.9"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )
    
    args = parser.parse_args()
    infer(args)