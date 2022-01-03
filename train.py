import argparse
import os
import torch
import numpy as np
from utils import set_seed, make_ner_data, compute_metrics
from dataset import NerDataset
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from transformers import RobertaForTokenClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train(args):
    #Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, 
                          do_lower_case=False,
                          unk_token='<unk>',
                          sep_token='</s>',
                          pad_token='<pad>',
                          cls_token='<s>',
                          mask_token='<mask>',
                         )
    #Build dataloader
    entity_label_list, train_data = make_ner_data(args.train_file, tokenizer)
    #sampelr 처리 못함
    kwargs = (
        {"num_workers":torch.cuda.device_count(), "pin_memory":True} if torch.cuda.is_available()
        else {}
    )
    train_dataset = NerDataset(tokenizer, 
                         train_data, 
                         entity_label_list,
                         args.max_length,
                         args.batch_size,
                         shuffle=False,
                         **kwargs
                        )
    train_dataloader = train_dataset.loader
    train_steps = len(train_dataloader) * args.epoch
    
    #Load model
    model = RobertaForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=len(entity_label_list),
        id2label = {str(i): label for i,label in enumerate(entity_label_list)},
        label2id = {label: i for i, label in enumerate(entity_label_list)}
    )
    model = model.cuda()
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        eps = 1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(args.warmup_ratio*train_steps),
        num_training_steps=train_steps
    )
    
    test_model = None
    best_f1_score = -1
    mb = master_bar(range(int(args.epoch)))
    for epoch in mb:
        print(f"epoch : {epoch+1}/{args.epoch}")
        train_loss = 0
        model.train()
        
        pb = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(pb):
            #initiate gradient
            model.zero_grad()
            
            batch = tuple(feature.to(device) for feature in batch)
            input_ids, attention_mask, labels = batch
            
            #forward
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            #loss
            loss = outputs.loss
            train_loss += loss.item()
            
            #backward / calculate gradient
            loss.backward()
            #limit gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            #weight update
            optimizer.step()
            
            #change lr
            scheduler.step()
        
        train_loss = train_loss / len(train_dataloader)
        print(f"epoch : {epoch+1}, loss : {train_loss}")
        
        f1_score, candidate_model = evaluate(args,
                                             model=model,
                                             tokenizer=tokenizer,
                                             eval_type='dev')
        if best_f1_score < f1_score:
            best_f1_score = f1_score
            best_model = candidate_model
        
    #evaluation이 가장 좋은 model ckpt 저장
    try:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    except OSError:
        print('Error: Creating directory : ' + args.save_path)
    #save best model
    best_model.save_pretrained(args.save_path)
    
    test_f1_score, _ = evaluate(args,
                                model=best_model,
                                tokenizer=tokenizer,
                                eval_type='test')
        

def evaluate(args, model, tokenizer, eval_type='dev'): #dev, test
    #Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Set mode
    model.eval()
    
    if eval_type=='dev':
        data_file = args.dev_file
        print("\nevaluation step\n")
    elif eval_type=='test':
        data_file = args.test_file
        print("\ntest step\n")
    
    #Build dataloader
    entity_label_list, data = make_ner_data(data_file, tokenizer)
    id2label = {str(i):label for i, label in enumerate(entity_label_list)}
    
    #sampelr 처리 못함
    kwargs = (
        {"num_workers": torch.cuda.device_count(), "pin_memory":True} if torch.cuda.is_available()
        else {}
    )
    dataset = NerDataset(tokenizer, 
                         data, 
                         entity_label_list,
                         args.max_length,
                         args.batch_size,
                         shuffle=False,
                         **kwargs
                        )
    dataloader = dataset.loader
    
    preds_cpu = None #prediction
    labels_cpu = None #answer
    
    for batch in dataloader:
        batch = tuple(feature.to(device) for feature in batch)
        input_ids, attention_mask, labels = batch
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask
                          )
        
        #get logit
        logits = outputs.logits
        
        #1. move logit and label to cpu to calculate metric
        #2. merge things to calculate
        if preds_cpu is None:
            preds_cpu = logits.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
        else:# (batch,seq_len,class), (batch, seq_len)
            preds_cpu = np.append(preds_cpu, logits.detach().cpu().numpy(), axis=0)
            labels_cpu = np.append(labels_cpu, labels.detach().cpu().numpy(), axis=0)
    preds_cpu = np.argmax(preds_cpu, axis=2) #(batch, seq_len)

    batch_num = preds_cpu.shape[0]
    seq_len = preds_cpu.shape[1]
    preds_final = [[] for _ in range(batch_num)]
    labels_final =  [[] for _ in range(batch_num)]

    def is_special_token(token):
        if token == -100:
            return True
        else:
            return False

    for batch in range(batch_num):
        for token in range(seq_len):
            label_id = labels_cpu[batch][token]
            pred_id = preds_cpu[batch][token]
            if not is_special_token(label_id):
                labels_final[batch].append(id2label[str(label_id)])
                preds_final[batch].append(id2label[str(pred_id)])

    f1_score = compute_metrics(labels_final, preds_final)
    print(f"[{eval_type}] f1_score : {f1_score}")
    return f1_score[0], model
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--model_path", 
        type=str, 
        default='/data/bowon_ko/TRoBERTa_BASE/210727/pretrain/checkpoint-2000000/'
    )
    
    parser.add_argument(
        "--save_path", 
        type=str, 
        default='/data/bowon_ko/TRoBERTa_BASE/220103/finetune/ner/'
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer/model/wordpiece_mecab/version_1.9"
    )
        
    parser.add_argument(
        "--train_file",
        type=str,
        default="jupyter/data/version0.4/train.tsv"
    )
    
    parser.add_argument(
        "--dev_file",
        type=str,
        default="jupyter/data/version0.4/val.tsv"
    )
    
    parser.add_argument(
        "--test_file",
        type=str,
        default="jupyter/data/version0.4/test.tsv"
    )
    
    parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="input batch size for train",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )
    
    parser.add_argument(
        "--epoch",
        type=int,
        default=10
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    
    args = parser.parse_args()
    set_seed(args)
    train(args)
    
    
    
        