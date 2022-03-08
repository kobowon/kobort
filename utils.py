import random
import logging
import datetime
import csv

import torch
import numpy as np
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

import transformers
from tokenizer import KobortTokenizer
import os
import re
import json
from tqdm import tqdm

from collections import defaultdict
import sys
from konlpy.tag import Mecab
import pandas as pd
import random



logger = logging.getLogger(__name__)

#dummy file 을 split해서 train, dev, test file로 변환
#input : dummy file
#output : train.tsv, dev.tsv, test.tsv

# regex를 통해 raw_text문장 내에서 tagging된 entity의 index를 뽑아내는 코드
def index_parser(raw_text, tagged_text):
    a = re.finditer('<.+?(:(PER|LOC|ORG|POH|DAT|TIM|DUR|MNY|PNT|NOH)){1}>', tagged_text)
    count = 0
    data = {}
    entity_data_list = []
    for i in a:
        entity_data = {}
        entity_data["start"] = i.span()[0] - count * 6
        entity_data["end"] = i.span()[1] - 6 - count * 6
        entity_data["text"] = i.group()[1:-5]
        entity_data["category"] = i.group()[-4:-1]
        if (raw_text[i.span()[0] - count * 6:i.span()[1] - 6 - count * 6] == i.group()[1:-5]):
            entity_data_list.append(entity_data)
        count += 1
    data["raw_text"] = raw_text
    data["entity_data"] = entity_data_list
    return data

def kmou_ner_parser():
    rawdata_dir = "./jupyter/rawdata/KMOU-NER-DATA/"
    dir_list = os.listdir(rawdata_dir)
    dir_list.sort()
    with open(rawdata_dir + "/NER_wholedata.txt", "w") as w:
        for file_name in tqdm(dir_list[1:]):
            if (file_name[-7:] == 'NER.txt' or file_name[:11] == 'EXOBRAIN_NE'):
                with open(rawdata_dir + file_name, "r") as f:
                    line = f.readline()
                    while line:
                        if line[0:2] == "##" and len(line[3:].strip()) > 10:
                            temp = line[3:].strip()
                            line = f.readline()
                            if line[0:2] == "##" and len(line[3:].strip()) > 10:
                                temp = temp + "\t" + line[3:].strip()
                                w.write(temp + "\n")
                        line = f.readline()

    with open(rawdata_dir + "/NER_wholedata.txt", "r") as f:
        lines = f.readlines()
        whole_data = {}
        whole_data["whole_data"] = []
        for line in lines:
            whole_data["whole_data"].append((index_parser(line.split("\t")[0], line.split("\t")[1])))

    with open("./jupyter/parsed_data/NER_wholedata_parsed.json", "w") as json_file:
        json.dump(whole_data, json_file)

    os.remove(rawdata_dir + "/NER_wholedata.txt")

class BioTagging(object):
    def __init__(self, label_path = None):
        if label_path and os.path.isfile(label_path):
            with open(label_path,"r") as f:
                self.bio_label = json.load(f) 
        else:
            self.bio_label = dict()
            self.bio_label['O'] = 0
            self.bio_cnt = 1
        
        self.label_cnt = defaultdict(lambda : defaultdict(int))
        self.label_type = defaultdict(list)
        self.data_idx = 0
        
        
    def entity_reindexing(self, text, n_text, entity):
        start, cnt, new_entities = 0, 0, []
        while entity:
            ent = entity.pop(0)
            if ent['text'][0] == ' ':
                ent['text'], ent['start'] = ent['text'].lstrip(), ent['start'] + 1
            if ent['text'][-1] == ' ':
                ent['text'], ent['end'] = ent['text'].rstrip(), ent['end'] - 1


            for i in range(start, len(text)):
                while text[i] != n_text[i + cnt]:
                    cnt += 1
                if i == ent['start']:
                    new_start = i + cnt
                if i == ent['end'] - 1:
                    ent['start'], ent['end'] = new_start, i + cnt + 1
                    new_entities.append(ent)
                    start = i+1
                    break
        return {'text' : n_text, 'entity' : new_entities}
    
        
    def bio_tagging(self, text, tok, entities, 
                    tok_encode, filt, unk):
        
        renew_entity = self._data_preprocess(text, entities)
        filt = set(filt)
        filt.add(' ')
        
        bio_tagging, cnt = [0 for i in range(len(tok))], 0
        #bio_tagging, cnt = ['O' for i in range(len(tok))], 0
        
        tok_idxs, tmp = [0, 0], []

        while renew_entity:
            entity, bio_idxs = renew_entity.pop(0), defaultdict(int)
            
            if 'B-' + entity['category'] not in self.bio_label:
                self._labeling(entity['category'])        

            tok_idxs, cnt = self._tagging_single_entity(tok_idxs, tok, entity, 
                                                        bio_idxs, cnt, filt, unk, 
                                                        text.replace(' ', '') )
            
            if cnt == 'Delete':
                return 0
            
            for check, (key, value) in enumerate(bio_idxs.items()):
                if len(tok[key]) != value:
                    return 0
                if not check:
                    tmp.append(entity['category'])
                
                bio_tagging[key] = self.bio_label['I-'+entity['category']] if check else self.bio_label['B-'+entity['category']]
                #bio_tagging[key] = 'I-'+entity['category'] if check else 'B-'+entity['category']
                
        for lt in tmp:
            self.label_cnt[self.data_idx][lt] += 1
            self.label_type[lt]
        self.data_idx += 1

        return {'text' : text, 'tokenize' : tok, 
                'bio_tagging' : bio_tagging, 'tok_encode' : tok_encode}
    
    
    def _tagging_single_entity(self, tok_idxs, tok, 
                               entity, bio_idxs, cnt, 
                               filt, unk, text):
        
        for ti in range(tok_idxs[0], len(tok)):
            if ti < tok_idxs[0]:
                continue

            if tok[ti] == unk:
                res = self._unk_preprocess(tok[ti:], text[cnt:], 
                                           filt, unk, ti)
                cnt += res[0]
                
                if cnt > entity['end'] or (cnt-res[0] < entity['start'] and cnt >= entity['start']):
                    return None, 'Delete'
               
                elif cnt >= entity['start'] and cnt <= entity['end']:
                    for unk_idx in range(ti, res[1]+1):
                        bio_idxs[unk_idx] += len(unk)
                    if cnt == entity['end']:
                        return [res[1]+1, 0], cnt + 1            
                    
                cnt += 1
                tok_idxs = [res[1] + 1, 0]
                continue
                                     
            for ci in range(tok_idxs[1], len(tok[ti])):
                if cnt >= entity['start'] and cnt <= entity['end']:
                    bio_idxs[ti] += 1

                    if tok[ti][ci] in filt:
                        continue
                    if cnt == entity['end']:
                        return [ti, ci], cnt

                elif tok[ti][ci] in filt:
                    continue
                cnt += 1
            tok_idxs[1] = 0
                
    def _unk_preprocess(self, remaining_tok, 
                        remaining_text, 
                        filts, unk, ti):
        
        if len(remaining_tok) == 1:
            return len(remaining_text) - 1, ti

        nxt_none_unk, cnt = remaining_tok[1], 0

        if nxt_none_unk == unk:
            res = self._unk_preprocess(remaining_tok[1:], 
                                       remaining_text, 
                                       filts, unk, ti+1)
            cnt += res[0]
            ti = max(ti, res[1])
        else:
            for filt in filts:
                nxt_none_unk = nxt_none_unk.replace(filt, '')

            tmp = len(nxt_none_unk)
            for i in range(len(remaining_text)):
                if remaining_text[i : i + tmp] == nxt_none_unk:
                    cnt = i - 1
                    break
        return cnt, ti

    def _data_preprocess(self, text, entities):
        renew_entity, start, cnt = [], 0, 0
        while entities:
            entity = entities.pop(0)
            for i in range(start, max(len(text), entity['end']+1)):
                if i == entity['start']:
                    entity['start'] = cnt
                elif i == entity['end']:
                    entity['end'], start = cnt - 1, i
                    renew_entity.append(entity)
                    break
                elif text[i] == ' ':
                    continue
                cnt += 1
        return renew_entity
        
    def _labeling(self, category):
        self.bio_label['B-' + category] = self.bio_cnt  
        self.bio_label['I-' + category] = self.bio_cnt+1
        self.bio_cnt += 2

    
    
    
class RestoreOutput(object):
    def __init__(self, label_path = './jupyter/bio_label_info.json'):
        with open(label_path,"r") as f:
            self.bio_label = json.load(f) 
        self.bio_label = {v : k[-3:] for k, v in self.bio_label.items()}
        
    def restore_output(self, text, bio_tagging, tokenize, filt, unk = '<unk>'):
        filt = set(filt)
        filt.add(' ')
        entities = self._get_entity_idx(text, bio_tagging, tokenize, filt, unk)
        final_output, start, char_cnt = "", 0, 0
        while entities:
            ent_type, ent_idx = entities.pop(0)     
            for i in range(start, len(text)):
                final_output += text[i]

                if text[i] == ' ':
                    char_cnt -= 1

                if char_cnt == ent_idx:
                    final_output += f'[{self.bio_label[ent_type]}]'
                    start = i + 1
                    char_cnt += 1
                    break
                char_cnt += 1
        return final_output + text[start:]


    def _get_entity_idx(self, text, bio_tagging, tokenize, filt, unk):
        entities, only_char = [], text.replace(' ', '')
        char_cnt = 0
        while tokenize:
            tok, bt = tokenize.pop(0), bio_tagging.pop(0)
            tmp = self._process_unk(tokenize, bio_tagging, only_char, filt, 
                                    unk) if tok == unk else sum([1 for _ in tok if _ not in filt])
            if bt:
                while bio_tagging and bio_tagging[0] == bt + 1:
                    tok = tokenize.pop(0)
                    bio_tagging.pop(0)
                    tmp += self._process_unk(tokenize, bio_tagging, only_char[tmp:], 
                                             filt, unk) if tok == unk else sum([1 for _ in tok if _ not in filt])
                entities.append([bt, char_cnt + tmp - 1])
            char_cnt += tmp
            only_char = only_char[tmp:]

        return entities

    def _process_unk(self, remaining_tok, bio_tagging, 
                     remaining_text, filts, unk):
        if not remaining_tok:
            return len(remaining_text)
        cnt = 0
        if remaining_tok[0] == unk:
            remaining_tok.pop(0)
            bio_tagging.pop(0)
            cnt += self._process_unk(remaining_tok, bio_tagging, 
                                     remaining_text, filts, unk)
        else:
            nxt_none_unk = remaining_tok[0]
            for filt in filts:
                nxt_none_unk = nxt_none_unk.replace(filt, '')
            tmp = len(nxt_none_unk)
            for i in range(len(remaining_text)):
                if remaining_text[i:i+tmp] == nxt_none_unk:
                    cnt = i
                    break
        return cnt        
    

def split_data(df, ratio, labels, get_val=True):
    data_idxs = [i for i in range(len(df))]
    random.Random(7).shuffle(data_idxs)
    tcnt = 2 if get_val else 1
    data = []
    while tcnt > 0:
        cur_data_type = 'test' if tcnt == 1 else 'validation'
        test_cnt = df[labels].sum().apply(lambda x: int(x * ratio))
        test_check = np.array(list(test_cnt.values))
        cur_test_idxs = set()
        while data_idxs:
            data_idx = data_idxs.pop(0)
            cur_idx_label_values = tuple(df.iloc[data_idx][labels].values)
            test_check -= cur_idx_label_values
            cur_test_idxs.add(data_idx)
            if not any(np.where(test_check <= 0, False, True)):
                for i in range(len(labels)):
                    print(f"# of Addtional {cur_data_type} From {labels[i]} == {abs(test_check[i])}")
                if get_val:
                    print('-' * 100)
                data.insert(0, [cur_data_type, df.iloc[list(cur_test_idxs)]])
                break
        tcnt -= 1
    data.insert(0, ['train', df.iloc[data_idxs]])
    return data


def DATA_processor(args):
    tokenizer = KobortTokenizer(model_name=args.tokenizer_type)
    whole_data = defaultdict(list)
    unsuit_data = defaultdict(list)
    bt = BioTagging(label_path=args.label_info_path)
    count_unsuitable, count_suitable = 0, 0
    
    with open(args.parsed_rawdata_path, 'r') as f:
        data = json.load(f)
    
    rg = len(data['whole_data'])
    for k in tqdm(range(rg)):
        d = data['whole_data'][k]
        text = d['raw_text']
        entity = d['entity_data']
        entity_analysis_purpose = entity[:]
        
        if k in [1420, 548]: # 데이터 자체에서 오류 발견
            continue
            
        # text안에 tokenizer가 사용하는 special token이 포함될 경우 학습데이터에서 제거
        flag =0
        for filt in args.filters:
            if filt in text:
                flag = 1
        if flag:
            continue
        
        # text를 형태소로 분절 및 결합
        if 'mecab' in args.tokenizer_type:
            tok_with_unk, tok_without_unk, morphs_text = tokenizer.tokenize(text)
            morph_res = bt.entity_reindexing(text, morphs_text, entity)
            text, entity = morph_res['text'], morph_res['entity']
        else:
            tok_with_unk, tok_without_unk = tokenizer.tokenize(text)
        
        tok_encode = tokenizer.encode(text)
        
        assert len(tok_without_unk) + 2 == len(tok_encode)
        if len(tok_without_unk) + 2 == len(tok_encode) and len(tok_encode) < 512:
            res = bt.bio_tagging(text, tok_without_unk, 
                                 entity, tok_encode, 
                                 args.filters,
                                 args.unknown_token)
            if res:
                count_suitable += 1
                for key, value in res.items():
                    if key == 'bio_tagging':
                        value = [0] + value + [0]
                    whole_data[key].append(value)
            else:
                unsuit_data['text'].append(text)
                unsuit_data['tokenize'].append(tok_without_unk)
                unsuit_data['entities'].append([entities['text'] for entities in entity_analysis_purpose])
                count_unsuitable += 1

        else:
            count_unsuitable += 1
    
    unsuit_data = pd.DataFrame(unsuit_data)
    unsuit_data.to_csv('unsuit_data.csv', index = False)
    
    print("unsuitable_data count : {}".format(count_unsuitable))
    print("suitable_data count : {}".format(count_suitable))
    
    for label_type in bt.label_type:
        bt.label_type[label_type] = [0 for _ in range(count_suitable)]

    for k, v in bt.label_cnt.items():
        for key, value in v.items():
            bt.label_type[key][k] = value

    labels = []
    for key, values in bt.label_type.items():
        labels.append(key)
        whole_data[key] = values

    if not os.path.exists(args.data_dir + '/parsed_data/'):
        os.makedirs(args.data_dir + '/parsed_data/')

    with open(args.data_dir +'/parsed_data/bio_label_info.json', 'w') as w:
        json.dump(bt.bio_label, w)

        
    whole_data = pd.DataFrame(whole_data)
    datas = split_data(whole_data, args.split_ratio, labels, get_val = False)
    
    
    for data_purpose, df in datas:
        for feature, p in [['tok_encode','sentence'], ['bio_tagging','label']]:
            torch.save(list(df[feature].values), args.data_dir + f'/gee_{p}_dataset_{data_purpose}.pt')
            
    print("Data Preprocessing Completed.")  




#ner 파일을 읽고, 라벨 리스트, 문장, 라벨 반환
def make_ner_data(file_path: str, tokenizer: KobortTokenizer, texts = None):
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
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = csv.reader(f, delimiter='\t')
            for i, line in enumerate(lines):
                if i > 0:
                    _, _, tokenized_text_with_unk, label_str = line
                    dataset = {
                        "tokenized_text_with_unk" : tokenized_text_with_unk,
                        "label_str" : label_str
                    }
                    dataset_list.append(dataset)
    else:#inference 시에 입력이 들어올 때
        for text in texts:
            tokenizer = KobortTokenizer("wp-mecab")
            tokens_with_unk, tokens_without_unk, _ = tokenizer.tokenize(text)
            tokenized_text_without_unk = ' '.join(tokens_without_unk)
            tokenized_text_with_unk = ' '.join(tokens_with_unk)
            dataset = {
                "tokenized_text" : tokenized_text_without_unk,
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
