import os
import argparse


#import torch
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=71, type=int, help='seed')

    ## Constant Variables
    
    parser.add_argument('--data_dir', default= "./data", type=str)
    parser.add_argument('--rawdata_dir', default= "./data/rawdata/", type=str)
    parser.add_argument('--parsed_rawdata_path', default= "./data/parsed_data/NER_wholedata_parsed.json", type=str)
    parser.add_argument('--label_info_path', default= "./data/parsed_data/bio_label_info.json", type=str)
    parser.add_argument('--model_pt_dir', default= "./model/fine_tuned_models/", type=str)
    parser.add_argument('--pretrain_model_path', default= "./model/kobort_1.9/", type=str)
    
    
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--maxlen', default=512, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--learning_rate', default=4e-5, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=int)
    parser.add_argument('--summary_step', default=200, type=int)
    parser.add_argument('--adam_epsilon', default=1e-8, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--num_class', default=21, type=int)
    parser.add_argument('--tokenizer_pad_idx', default=1, type=int)
    parser.add_argument('--label_pad_idx', default=0, type=int)
    
    ## Mutable Variable
    # Hyperparameter
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int, choices=[1, 4, 8, 16, 32])
    parser.add_argument("--accumulated_steps", default=1, type=int, choices=[1, 2, 4, 8, 16])
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--loss_fn", default='cross_entropy', type=str, choices=['label_smoothing', 'cross_entropy'])
    parser.add_argument("--smoothing", default=0.3, type=float)
    
    
    # Model
    parser.add_argument("--model", default="TbertCRF", type=str, choices=["TbertLSTMCRF", "TbertCRF" ,"bertLSTM", "Tbert"])
    
    # Data
    parser.add_argument("--split_ratio", default=0.1, type=float)
    parser.add_argument("--get_val", default=True, choices=[True, False])
    parser.add_argument("--save_data_type", default='tsv', type=str, choices=['pt', 'tsv'])

    
    # Tokenizer
    parser.add_argument('--filters', type = list, nargs = '+', default=['#'])
    parser.add_argument('--unknown_token', default='<unk>', type=str)
    parser.add_argument("--tokenizer_model_name", default='wp-mecab', type=str, choices=['wp', 'wp-mecab'])
    
    # Fine-Tuned Model
    parser.add_argument('--fine_tuned_model', default= "TbertCRF_batch16_f1score0.9209.pt", type=str)
    
    args = parser.parse_args()
    return args

