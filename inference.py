import argparse
import os
import torch
import numpy as np
from utils import set_seed, make_ner_data, return_ner_tagged_sentence_v2
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
    infer_texts = ["오늘 고보원은 9시에 티맥스로 출근했다", "티맥스 소프트는 매각 되었다", "세련미 가미한 공격 우리은행 박혜진, 임영희, 양지희는 공격에도 능하다.", "쿠션 중앙의 작은 구멍으로 파운데이션이 나오는데, 양 조절이 익숙해지려면 몇 번 사용해봐야 할듯 해요.” ★★★★★원선희 “보통 쿠션 제품은 퍼프에 너무 많은 양이 묻어나 퍼프가 쉽게 더러워지고 양 조절이 힘든 단점이 있잖아요.","신카이 마코토 감독ㆍ배우 한예리··· ‘너의 이름은’ 메가토크에서 만나요!", '깐쇼새우 만드는 법, 이연복 셰프 비법은 "튀김옷 반죽에 식용류"']
    
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
        original_tokens = inference_data[batch_idx]["sentence"].split()
        tokens = ['<s>'] + original_tokens + ['</s>']
        print(f"\n문장 {batch_idx}의 결과는 아래와 같습니다")
        origin_tokens_labels = []
        #original_tokens의 실질적인 토큰 범위 내에서 (special token 제외)
        for token_idx in range(1,len(tokens)-1): #유효한 토큰들만 살펴보기 <pad> 제외
            #예측한 label id를 label로 변환
            pred_label_index = preds_cpu[batch_idx][token_idx]
            pred_label_str = id2label[str(pred_label_index)]
            origin_tokens_labels.append(pred_label_str)
            print(f'{token_idx}번째 토큰 "{tokens[token_idx]}"의 라벨은 "{pred_label_str}"입니다')
        print(f"\n원문장 : {origin_text}")
        print(f"토크나이즈된 문장 : {original_tokens}")
        print(f"토큰에 대한 라벨 리스트 : {origin_tokens_labels}") 
        
        ner_tagged_sentence, ner_tagged_id_list = return_ner_tagged_sentence_v2(origin_text, original_tokens,origin_tokens_labels)
  
        print(f'태깅 문장 : {ner_tagged_sentence}')
        for item in ner_tagged_id_list:
            start, end = item
            print(f'entity 확인 : {origin_text[start:end+1]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        default='/data/bowon_ko/TRoBERTa_BASE/210727/finetune/ner/'
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data/bowon_ko/wordpiece/version_1.9"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )
    
    args = parser.parse_args()
    infer(args)