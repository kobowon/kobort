from utils import BIO_processor as BIO
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        default='wp-mecab',
        choices = ['wp-mecab', 'wp']
    )
    
    parser.add_argument(
        "--make_file",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    BIO(args)
    