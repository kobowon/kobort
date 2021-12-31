import random
import logging
import datetime


import torch
import numpy as np
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

import transformers
from transformers import PreTrainedTokenizer
from tokenizer import make_tokens

logger = logging.getLogger(__name__)

#dummy file 을 split해서 train, dev, test file로 변환
#input : dummy file
#output : train.tsv, dev.tsv, test.tsv
def construct_ner_file(file_path:str):
    

#ner 파일을 읽고, 라벨 리스트, 문장, 라벨 반환
def make_ner_data(file_path: str, tokenizer: PreTrainedTokenizer, texts = None):
    #label_list 작성
    entity_label_list = ['O', 
         'B-DUR', 'I-DUR', 
         'B-DAT', 'I-DAT', 
         'B-LOC', 'I-LOC', 
         'B-MNY', 'I-MNY', 
         'B-TIM', 'I-TIM', 
         'B-ORG', 'I-ORG', 
         'B-PNT', 'I-PNT', 
         'B-POH', 'I-POH', 
         'B-NOH', 'I-NOH', 
         'B-PER', 'I-PER']
    
    #tokenization과정을 거치고 이를 space 단위로 결합하여 반환
    dataset_list = []
    if file_path: #학습할 때 혹은 inference할 파일이 있을 때
        with open(file_path, 'r', encoding='utf-8') as lines:
            for i, line in enumerate(lines):
                if i > 0:
                    _, _, sentence, label_str = line.strip().split('\t')
                    dataset = {
                        "sentence" : sentence,
                        "label_str" : label_str
                    }
                    dataset_list.append(dataset)
    else:#inference 시에 입력이 들어올 때
        for text in texts:
            tokens_with_unk, tokens_without_unk = make_tokens(text=text,model_name="wp-mecab")
            tokenized_text = ' '.join(tokens_without_unk)
            tokenized_text_with_unk = ' '.join(tokens_with_unk)
            dataset = {
                "tokenized_text" : tokenized_text,
                "tokenized_text_with_unk" : tokenized_text_with_unk,
                "label_str" : None
            }
            dataset_list.append(dataset)
                
    return entity_label_list, dataset_list  

'''
origin_text : 입력 문장
origin_tokens : tokenization 과정을 거쳤지만 <unk>인 경우에도 <unk>으로 변경 안 한 토큰
tokens : tokenization 과정을 거치고 <unk>인 경우를 <unk>으로 변경한 토큰들 (이전)
labels : origin_tokens, tokens에 대한 라벨
'''
def return_ner_tagged_sentence_plus(origin_text, origin_tokens, labels):
    ner_tagged_sentence = ''
    ner_tagged_id_list = []
    
    #divide label per character token
    #space를 제외한 모든 글자에 대한 라벨을 붙이기
    #abc : B-LOC → a : B-LOC, b : I-LOC, c : I-LOC
    char_labels = []
    char_tokens = []
    for token, label in zip(origin_tokens, labels):
        if label == 'O':
            for i in range(len(token)):
                #한글자씩 분해시킨 토큰
                char_token = token[i]
                char_label = 'O'
                if char_token != "#":
                    char_labels.append(char_label)
                    char_tokens.append(char_token)
        elif label.startswith("B-"):
            entity = label[2:]
            for i in range(len(token)):
                char_token = token[i]
                if char_token != "#":
                    if i == 0:
                        char_label = "B-"+ entity
                    elif i > 0:
                        char_label = "I-"+ entity
                elif char_token == "#":
                    if i < 2:
                        continue
                    elif i == 2:
                        char_label = "B-"+ entity
                    elif i > 2:
                        char_label = "I-"+ entity
                        
                char_labels.append(char_label)
                char_tokens.append(char_token)
        elif label.startswith("I-"):
            entity = label[2:]
            for i in range(len(token)):
                char_token = token[i]
                char_label = "I-"+entity
                if char_token != "#":
                    char_tokens.append(char_token)
                    char_labels.append(char_label)
        
    #merge char tokens
    text_len = len(origin_text)
    left = 0
    right = 0
    char_id = 0 #char_tokens, char_labels 를 스캔하는 id
    
    try:
        while right < text_len: #종료조건 
            #space가 아닌 char에 대해
            if origin_text[right] == ' ':
                ner_tagged_sentence += origin_text[right]
                right += 1
            elif origin_text[right] != ' ':
                #항상 origin_text[right] 와 char_tokens[char_id] 가 동일함
                if char_labels[char_id].startswith('I-'): #잘못된 예측 I가 먼저 나온 상황
                    ner_tagged_sentence += origin_text[right]
                    char_id += 1 #개체명으로 인식하지 않고 넘기기
                    right += 1

                elif char_labels[char_id] == 'O':
                    ner_tagged_sentence += origin_text[right]
                    char_id += 1
                    right += 1

                elif char_labels[char_id].startswith('B-'): #여기서는 작업마치고 left, right 저장해야함
                    entity = char_labels[char_id][2:]
                    pred_label = 'I-'+entity
                    left = right
                    right += 1
                    char_id += 1
                    while True:
                        if origin_text[right] == ' ':
                            right += 1
                        elif origin_text[right] != ' ':
                            if char_labels[char_id] == pred_label:
                                if char_id == len(char_labels)-1:
                                    ner_tagged_sentence += (origin_text[left:right+1]+'['+entity+']')
                                    ner_tagged_id_list.append([left,right,entity])
                                    right += 1
                                    break
                                else:
                                    char_id += 1
                                    right += 1
                            elif char_labels[char_id] != pred_label:
                                if origin_text[right-1] == ' ':
                                    ner_tagged_sentence += (origin_text[left:right-1]+'['+entity+']'+' ')
                                    ner_tagged_id_list.append([left,right-2,entity])
                                else:
                                    ner_tagged_sentence += (origin_text[left:right]+'['+entity+']')
                                    ner_tagged_id_list.append([left,right-1,entity])
                                left = right
                                break
    except:
        print('ERROR')
        print(origin_text)
        
    
    # for char_token, char_label in zip(char_tokens, char_labels):
    #     print(f"token : {char_token}, label : {char_label}")
    return ner_tagged_sentence, ner_tagged_id_list
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    
    result = {"precision":seqeval_metrics.precision_score(labels, preds, suffix=True),
              "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
              "f1": seqeval_metrics.f1_score(labels, preds, suffix=True)
    }
    return result["f1"], result
