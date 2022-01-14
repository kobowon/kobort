from utils import BIO_processor as BIO
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        default='wp',
        choices = ['wp-mecab', 'wp']
    )
    args = parser.parse_args()
    BIO(args)
    