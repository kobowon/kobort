from utils import DATA_processor as BIO
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', default= "./data", type=str)
    parser.add_argument('--rawdata_dir', default= "./data/rawdata/", type=str)
    parser.add_argument('--parsed_rawdata_path', default= "./data/parsed_data/NER_wholedata_parsed.json", type=str)
    parser.add_argument('--label_info_path', default= "./data/parsed_data/bio_label_info.json", type=str)
    
    # ETC
    parser.add_argument("--split_ratio", default=0.1, type=float)

    # Tokenizer
    parser.add_argument('--filters', type = list, nargs = '+', default=['#'])
    parser.add_argument('--unknown_token', default='<unk>', type=str)
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        default='wp',
        choices = ['wp-mecab', 'wp']
    )
    
    parser.add_argument(
        "--make_file",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    
    kmou_ner_parser(data_dir=args.data_dir)
    
    BIO(args)
    