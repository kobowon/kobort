import csv
from tokenizer import make_tokens
from tqdm import tqdm

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

answers = []
with open("jupyter/data/test.tsv", "r", encoding="utf-8") as f:
    lines = csv.reader(f, delimiter='\t',)
    for i, line in enumerate(lines):
        if i>0:
            #print(line)
            origin_text, tokens_str, label_str = line
            #morphed_text = make_tokens(origin_text, model_name="mecab")
            #tokens, origin_tokens = make_tokens(origin_text, model_name="wp-mecab")
            #tokens_text = ' '.join(tokens)
            #origin_tokens_text = ' '.join(origin_tokens)
            tokens = tokens_str.split()
            labels = label_str.split()
            tagged_sentence, tagged_id_list = return_ner_tagged_sentence_plus(origin_text, tokens, labels)

            
            answers.append([origin_text, tagged_sentence])

for i, answer in enumerate(answers):
    if i<2:
        print(answer[0], answer[1])

   
        
with open("jupyter/data/answer.tsv", "w", encoding="utf-8") as f:
    field_names = ["original_text", "ner_tagged_text"]
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(field_names)
    for answer in answers:
        writer.writerow([answer[0], answer[1]])

        