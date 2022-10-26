import os
import argparse

def create_directory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

def read_subwords(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        subwords = f.read().split('\n')
        #subwords에는 중복되는 서브워드가 없어야 함
        assert len(subwords) == len(list(set(subwords))) 
        return subwords
        
def write_vocab(vocab_path, vocab_num, result):
    with open(vocab_path, 'w', encoding='utf-8') as f:
        if len(result) > vocab_num:
            for i, w in enumerate(result[:vocab_num]):
                if i == vocab_num-1:
                    f.write(w)
                else:
                    f.write(w+'\n')
        else:
            for i, w in enumerate(result):
                if i == len(result)-1:
                    f.write(w)
                else:
                    f.write(w+'\n')
                    
def tune(args):
    subwords = read_subwords(args.wp_path+'vocab.txt')

    # with open('model/wordpiece/version_2.2/vocab.txt', 'r', encoding='utf-8') as f:
    #     subwords = f.read().split('\n')

    new_subwords = {}
    #subword 중에서 ♬가 포함된 subword 중 유의미한 경우만 잡아내고 나머지는 pass하기
    #현재 subword는 띄어쓰기 단위를 기준으로 만들어지며 subword에는 여러 종류가 있을 수 있지만
    #현 실험에서는 subword는 [조사+XSN] OR 그 외로 나눈다
    #1. ##♬[조사+XSN] : 이
    #2. ##♬[조사+XSN]♬[X]
    #3. [X] : 일반 subword
    #4. ##[X] : [조사+XSN]에 해당하지 않는 그 외 subword 

    #전제
    #♬뒤에는 반드시 mecab이 분리한 문자가 온다 : ♬[X] -> X가 ''일 수는 없다

    OPTION = True #32000개에서 +1 효과 있었음 (27541->27542)
    FLAG = True

    for subword in subwords: #항상 유의 : subword는 일반 문장에서 띄어쓰기로 구분이 된 대상으로부터 만들어짐
        if '♬' not in subword: #mecab을 통한 [조사+XSN]가 들어간 부분이 없다면 / 3,4번 경우
            if subword not in new_subwords: 
                new_subwords[subword] = 1

        elif '♬' in subword:
            if subword == '##♬' or subword == '♬':
                continue
            elif subword.startswith('##♬'): #1,2번 경우
                if '♬' not in subword[3:]: #1번
                    if FLAG:
                        new_subword = '##' + subword[3:]
                        if new_subword not in new_subwords:
                            new_subwords[new_subword] = 1
                            continue
                #제거할까하다가 넣음
                elif '♬' in subword[3:]: #2번  ##♬[X]♬[X]...
                    print(f'[DONE]"{subword}": ##♬[X]♬[X]... 형태의 subword입니다')
                    if OPTION:
                        candidates = subword.split('♬')
                        for c in candidates[1:]: # candidates에서 ##제외 22.8.26 수정
                            new_subword = '##' + c
                            if new_subword not in new_subwords:
                                new_subwords[new_subword] = 1

            else: #'♬[X]'('♬[X]♬[X]...'), '[X]♬[X]'('[X]♬[X]♬[X]...) 
                candidates = subword.split('♬')
                #'♬'가 두 번 이상나오면 '명사♬[X1]♬[X2]' 인 경우 [X1],[X2] 모두 조사 혹은 XSN임 왜냐하면 X1은 [조사+명사]일 수가 없음 조사 이후에는 띄어쓰기가 되므로 subword가 명사와 함께 구성될 수 없음 

                #제거할까하다가 넣음, 분리가 잘 안 된 경우
                if len(candidates) > 2:
                    print(f'[DONE]"{subword}": [X]♬[X]♬.. 혹은 ♬[X]♬[X].. 형태의 형태의 subword입니다')
                    if OPTION:
                        first_token = candidates[0]
                        after_tokens = candidates[1:]
                        #first token
                        if first_token == '': # ♬[X]♬[X]../ '♬[X]'로 시작하는 경우는 띄어쓰기+조사가 왔다는 건데 조사랑 앞에 체언이랑 붙여써서 말이 안 되는 경우 이므로 제거, XSN(접미사) 또한 앞말과 붙여씀
                            continue
                        else: #[X]♬[X]♬.., ##[X]♬[X]♬.. 이 경우
                            if first_token not in new_subwords:
                                new_subwords[first_token] = 1
                            #after token
                            for c in after_tokens:
                                new_subword = '##' + c
                                if new_subword not in new_subwords:
                                    new_subwords[new_subword] = 1
                            continue

                elif len(candidates) == 2: #'♬'가 두 번 나오는 경우는 명사♬조사 경우뿐, 왜냐하면 조사 뒤에는 보통 띄어쓰기가 들어가기 때문에 추가로 명사가 붙을 확률이 0%라고 봄
                    first_token = candidates[0]
                    sec_token = candidates[1] # '##'으로 시작할 수 없음 -> ♬[X] 이렇게 오는 경우에서 X가 조사(mecab에서 분리된 걸 조사라고 통칭함) 가 아닌 경우는 없음, 조사 다음에는 보통 띄어쓰기가 오기 때문 
                    #first_token
                    if first_token == '': #♬[X] 의 경우 , 띄어쓰기 + 조사 오는 경우 제거 : 22.08.26 policy
                        continue
                    else: # '[X]♬[X]' : first_token -> 체언, '##[X]♬[X]' : first_token -> 체언의 일부
                        if first_token not in new_subwords:
                            new_subwords[first_token] = 1
                    #sec_token
                    new_subword = '##' + sec_token
                    if new_subword not in new_subwords: 
                        new_subwords[new_subword] = 1

    result = list(new_subwords.keys())

    create_directory(args.vtuning_path)
    write_vocab(args.vtuning_path+'vocab.txt', args.vocab_num, result)
                
    print("vocab tuning complete! :)")
    subwords = read_subwords(args.vtuning_path+'vocab.txt')
    print(len(subwords))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--wp_path", type=str, default = 'model/wordpiece/version_2.2/')
    parser.add_argument("--vtuning_path", type=str, default = 'model/wordpiece/version_2.2.11/')
    parser.add_argument("--vocab_num", type=int, default = 32000)
    
    args = parser.parse_args()
    tune(args)