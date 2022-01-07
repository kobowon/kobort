"""
Preprocessing script before training the distilled model.
"""
import argparse
import logging
import pickle
from collections import Counter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)"
    )
    parser.add_argument(
        "--data_file", type=str, default="/data/bowon_ko/data/dump.bert.pickle", help="The binarized dataset."
    )
    parser.add_argument(
        "--token_counts_dump", type=str, default="/data/bowon_ko/data/token_counts.bert.pickle", help="The dump file."
    )
    parser.add_argument("--vocab_size", default=50000, type=int)
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_file}")
    with open(args.data_file, "rb") as fp:
        data = pickle.load(fp)

    logger.info("Counting occurences for MLM.")
    counter = Counter()
    for tk_ids in data: #하나의 np.int32(d) : [1 2 3]
        counter.update(tk_ids) #Counter({1: 1, 2: 1, 3: 1})
    counts = [0] * args.vocab_size
    for k, v in counter.items():
        counts[k] = v #k번째 단어의 횟수는 v

    logger.info(f"Dump to {args.token_counts_dump}")
    with open(args.token_counts_dump, "wb") as handle:
        pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)