from args import parse_args
from utils import DATA_processor as BIO
from utils import kmou_ner_parser

if __name__ == "__main__":
    kmou_ner_parser()
    args = parse_args()
    BIO(args)
    
    