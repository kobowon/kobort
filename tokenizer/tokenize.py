from transformers import BertTokenizer,AlbertTokenizer
from konlpy.tag import Mecab

wordpiece_info = {"vocab_path" : "tokenizer/model/wordpiece/version_2.4.1"}#1.4

wordpiece_mecab_info = {"vocab_path" : "tokenizer/model/wordpiece_mecab/version_1.9"}

sentencepiece_mecab_info = {"vocab_path" : "tokenizer/model/sentencepiece_mecab/version_0.1/version_0.1.model"}

class KobortTokenizer:
    def __init__(self, model_name):
        self.model_name = model_name
        if "mecab" in self.model_name:
            self.mecab = Mecab()
            
        if self.model_name == "sp-mecab":
            tokenizer_path = sentencepiece_mecab_info["vocab_path"]
            self.tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path,
                                                    do_lower_case=False,
                                                    unk_token='<unk>',
                                                    sep_token='</s>',
                                                    pad_token='<pad>',
                                                    cls_token='<s>',
                                                    mask_token='<mask>',
                                                    use_fast=True)
        elif "wp" in self.model_name:
            if self.model_name == "wp-mecab":
                tokenizer_path = wordpiece_mecab_info["vocab_path"]
            elif self.model_name == "wp":
                #tokenizer_path = wordpiece_info["vocab_path"]
                tokenizer_path = wordpiece_info["vocab_path"] #vocab이 형태소를 반영한 상태로 더 잘 되어 있을 거라 예상
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, 
                                                  do_lower_case=False,
                                                  unk_token='<unk>',
                                                  sep_token='</s>',
                                                  pad_token='<pad>',
                                                  cls_token='<s>',
                                                  mask_token='<mask>',
                                                  use_fast=True)
    def tokenize(self, text):
        if "mecab" in self.model_name:
            morphs = self.mecab.morphs(text)
            morphs_text = " ".join(morphs)
            tokens_with_unk, tokens_without_unk = self.tokenizer.tokenize(morphs_text)
            return tokens_with_unk, tokens_without_unk, morphs_text
        else:
            tokens_with_unk, tokens_without_unk = self.tokenizer.tokenize(text)
            return tokens_with_unk, tokens_without_unk

    def encode(self, text):
        return self.tokenizer.encode(text)


# def load_tokenizer(model_name="wp-mecab"):
#     if model_name == "sp-mecab":
#         tokenizer_path = sentencepiece_mecab_info["vocab_path"]
#         tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path,
#                                                 do_lower_case=False,
#                                                 unk_token='<unk>',
#                                                 sep_token='</s>',
#                                                 pad_token='<pad>',
#                                                 cls_token='<s>',
#                                                 mask_token='<mask>',
#                                                 use_fast=True)
#     elif "wp" in model_name:
#         if model_name == "wp-mecab":
#             tokenizer_path = wordpiece_mecab_info["vocab_path"]
#         elif model_name == "wp":
#             tokenizer_path = wordpiece_info["vocab_path"]
#         tokenizer = BertTokenizer.from_pretrained(tokenizer_path, 
#                                               do_lower_case=False,
#                                               unk_token='<unk>',
#                                               sep_token='</s>',
#                                               pad_token='<pad>',
#                                               cls_token='<s>',
#                                               mask_token='<mask>',
#                                               use_fast=True)
#     return tokenizer

    



# def make_tokens(tokenizer, text, model_name="wp-mecab"):
#     if "mecab" in model_name:
#         mecab = Mecab()
#         #text를 형태소로 분절 및 결합
#         morphs = mecab.morphs(text)
#         text = " ".join(morphs)
        
#     #텍스트를 토크나이즈
#     tokens, origin_tokens = tokenizer.tokenize(text)
    
#     if "mecab" in model_name:
#         return tokens, origin_tokens, text
    
#     return tokens, origin_tokens
    