#wordpiece 학습
import argparse
import json # import json module
import io
import os 
from tokenizers import SentencePieceBPETokenizer, BertWordPieceTokenizer

def create_directory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

#pretrain_data_path : #전처리된 데이터 (J, XSN 앞에 ♬가 추가된 데이터) 
def train(args):
    if args.tokenizer_type == "wp":
        tokenizer = BertWordPieceTokenizer(strip_accents=False, #cased model인 경우 False
                                           lowercase=False)
    elif args.tokenizer_type == "sp":
        tokenizer = SentencePieceBPETokenizer()
    else:
        assert('select right tokenizer')

    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>"]

    unk_ids = 10
    unk_special_token_list = ['[UNK{}]'.format(i) for i in range(unk_ids)]
    special_tokens.extend(unk_special_token_list)

    unused_ids = 200
    unused_special_token_list = ['[unused{}]'.format(i) for i in range(unused_ids)]
    special_tokens.extend(unused_special_token_list)

    tokenizer.train(
        files=[args.data_path],
        vocab_size=args.vocab_size,
        min_frequency=1,#vocab이 크면 올려줘야 할 거 같음, vocab이 작으면 문제 없을 듯 
        limit_alphabet=6000,
        show_progress=True,
        special_tokens=special_tokens,
    )

    create_directory(args.vocab_dir)
    vocab_path = args.vocab_dir + "vocab.json"
    tokenizer.save(vocab_path)                                                                                                 

    vocab_file = args.vocab_dir+'vocab.txt'
    f = io.open(vocab_file,'w',encoding='utf-8')
    with open(vocab_path) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')
        f.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_path",
        type=str,
        default = None
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="wp"
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None
    )
    
    args=parser.parse_args()
    train(args)