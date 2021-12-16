from transformers import BertTokenizer,AlbertTokenizer
from konlpy.tag import Mecab

wordpiece_info = {"vocab_path" : "tokenizer/model/wordpiece/version_1.4"}

wordpiece_mecab_info = {"vocab_path" : "tokenizer/model/wordpiece_mecab/version_1.9"}

sentencepiece_mecab_info = {"vocab_path" : "tokenizer/model/sentencepiece_mecab/version_0.1/version_0.1.model"}


def make_tokens(text, model_name="wp-mecab"):
    if model_name == "sp-mecab":
        tokenizer_path = sentencepiece_mecab_info["vocab_path"]
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path,
                                                do_lower_case=False,
                                                unk_token='<unk>',
                                                sep_token='</s>',
                                                pad_token='<pad>',
                                                cls_token='<s>',
                                                mask_token='<mask>',
                                                use_fast=True)
    elif "wp" in model_name:
        if model_name == "wp-mecab":
            tokenizer_path = wordpiece_mecab_info["vocab_path"]
        elif model_name == "wp":
            tokenizer_path = wordpiece_info["vocab_path"]
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, 
                                              do_lower_case=False,
                                              unk_token='<unk>',
                                              sep_token='</s>',
                                              pad_token='<pad>',
                                              cls_token='<s>',
                                              mask_token='<mask>',
                                              use_fast=True)
    
    if "mecab" in model_name:
        mecab = Mecab()
        #text를 형태소로 분절 및 결합
        morphs = mecab.morphs(text)
        text = " ".join(morphs)
        
    #텍스트를 토크나이즈
    tokens = tokenizer.tokenize(text)
    
    return tokens
    