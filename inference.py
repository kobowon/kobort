import argparse
import os
import torch
import numpy as np
from utils import set_seed, make_ner_data
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
    infer_texts = ["오늘 고보원은 9시에 티맥스로 출근했다", "티맥스 소프트는 매각 되었다"]
    
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
        #원래 문장 <s> </s> 포함 안 된 문장
        original_tokens = inference_data[batch_idx]["sentence"].split()
        original_tokens = ['<s>'] + original_tokens + ['</s>']
        print(f"\n문장 {batch_idx}의 결과는 아래와 같습니다")
        for token_idx in range(len(original_tokens)): #유효한 토큰들만 살펴보기 <pad> 제외
            #예측한 label id를 label로 변환
            pred_label_index = preds_cpu[batch_idx][token_idx]
            pred_label_str = id2label[str(pred_label_index)]           
            print(f'{token_idx}번째 토큰 "{original_tokens[token_idx]}"의 라벨은 "{pred_label_str}"입니다')   

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