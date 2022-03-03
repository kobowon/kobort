import os
def create_directory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

with open('model/wordpiece/version_2.4/vocab.txt', 'r', encoding='utf-8') as f:
    subwords = f.read().split('\n')

#subwords에는 중복되는 서브워드가 없어야 함
assert len(subwords) == len(list(set(subwords))) 
new_subwords = []
check_double = {}
for subword in subwords:
    if '♬' in subword:
        if subword == '♬':
            pass
        elif subword[0] == '♬' and ('♬' not in subword[1:]):  #두 글자 이상의 ♬가 들어가 서브워드에 대해서
            #특수 문자를 이제 wordpiece 구분자인 ##으로 변경 (어짜피 조사나 XSN은 명사 뒤에 붙는 말이기 때문에 ##붙음)
            new_subword = '##'+subword[1:]
            #new_subwords.append(new_subword)
            if new_subword in check_double:
                #mecab 조사 분리가 잘 되지 않아서 ##조사가 나올 수도 있음
                print('[1]이미 있는 단어 : ',new_subword)
            else:#처음 등장하는 단어라면
                new_subwords.append(new_subword) #단어 넣고
                check_double[new_subword] = 1#이제는 있는 단어라고 표시
        else:
            pass
    else:
        if subword in check_double:
            #♬로 시작하는 서브워드가 ##서브워드로 변경되어 중복되는 경우 (♬서브워드의 우선순위를 높임)
            print('[2]이미 있는 단어 : ',subword)
        else:
            new_subwords.append(subword)#단어 넣고
            check_double[subword] = 1 #이제는 있는 단어라고 표시

vocab_path = 'model/wordpiece/version_2.4.1/'
create_directory(vocab_path)

with open(vocab_path+'vocab.txt', 'w', encoding='utf-8') as f:
    for w in new_subwords:
        f.write(w+'\n')
print("vocab tuning complete! :)")
print(len(new_subwords))