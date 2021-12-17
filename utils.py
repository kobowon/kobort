import random
import logging
import datetime


import torch
import numpy as np
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

import transformers
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

#ner 데이터를 읽고, 라벨 리스트, 문장, 라벨 반환
def make_ner_data(file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
    #label_list 작성
    entity_label_list = 
        ['O', 
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
    

    dataset_list = []
    with open(file_path, 'r', encoding='utf-8') as lines:
        for i, line in enumerate(lines):
            if i > 0:
                _, _, sentence, label_str = line.strip().split('\t')
                dataset = {
                    "sentence" : sentence,
                    "label_str" : label_str
                }
                dataset_list.append(dataset)
                
    return entity_label_list, dataset_list
    
    
def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def format_time(elapsed): 
    #반올림
    elapsed_rounded = int(round((elapsed)))
    
    #hh:mm 형태로 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))    

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
