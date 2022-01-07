"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time

import numpy as np

from transformers import BertTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--file_path", type=str, default="/data/bowon_ko/data/preprocess_completion/pretrain_v1.10.txt", help="The path to the data.")
    parser.add_argument("--tokenizer_type", type=str, default="bert", choices=["bert", "roberta"])
    parser.add_argument("--tokenizer_path", type=str, default="/data/bowon_ko/wordpiece/version_1.9", help="The tokenizer path to use.")
    parser.add_argument("--dump_file", type=str, default="/data/bowon_ko/data/distil/dump", help="The dump file prefix.")
    args = parser.parse_args()

    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, 
                                                      do_lower_case=False,
                                                      unk_token='<unk>',
                                                      sep_token='</s>',
                                                      pad_token='<pad>',
                                                      cls_token='<s>',
                                                      mask_token='<mask>',
                                                      use_fast=True)
    elif args.tokenizer_type == "roberta":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, 
                                                      do_lower_case=False,
                                                      unk_token='<unk>',
                                                      sep_token='</s>',
                                                      pad_token='<pad>',
                                                      cls_token='<s>',
                                                      mask_token='<mask>',
                                                      use_fast=True)
    bos = tokenizer.special_tokens_map["cls_token"]
    sep = tokenizer.special_tokens_map["sep_token"]
    

    logger.info(f"Loading text from {args.file_path}")
    with open(args.file_path, "r", encoding="utf8") as fp:
        data = fp.readlines()

    logger.info("Start encoding")
    logger.info(f"{len(data)} examples to process.")

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()
    for text in data:
        text = f"{bos} {text.strip()} {sep}"
        #token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ㆍ정의화 number일 직권상정여 쟁점법안 연계 처리 불투명"))
        #print(token_ids)
        #return
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rslt.append(token_ids)

        iter += 1
        if iter % interval == 0:
            end = time.time()
            logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
            start = time.time()
    logger.info("Finished binarization")
    logger.info(f"{len(data)} examples processed.")
    

    vocab_size = tokenizer.vocab_size
    rslt_ = []
    if vocab_size < (1 << 16): #2의 16승
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]

    random.shuffle(rslt_)
    
    dp_file = f"{args.dump_file}.{args.tokenizer_type}.pickle"
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()