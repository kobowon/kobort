토크나이저 사용 예제
```
from tokenizer import KobortTokenizer
tokenizer = KobortTokenizer("wp-mecab")
tokenizer.tokenize("보원이는 밥을 먹었다")
```
#NER 파인튜닝 예제

1.데이터 생성 작업
```
python data_processing.py \
    --tokenizer_type wp-mecab \
    --make_file True
```

2.파인튜닝 
```
python train.py \
    --batch_size 16 \
    --max_length 512
```