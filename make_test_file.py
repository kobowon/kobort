import csv
from tokenizer import make_tokens
from tqdm import tqdm

answers = []
with open("data/whole_data.tsv", "r", encoding="utf-8") as f:
    lines = csv.reader(f, delimiter='\t',)
    for i, line in tqdm(enumerate(lines)):
        if i>0:
            #print(line)
            origin_text, entity = line
            morphed_text = make_tokens(origin_text, model_name="mecab")
            tokens, origin_tokens = make_tokens(origin_text, model_name="wp-mecab")
            tokens_text = ' '.join(tokens)
            origin_tokens_text = ' '.join(origin_tokens)
            answers.append([origin_text, morphed_text, tokens_text, origin_tokens_text, entity])

for i, answer in enumerate(answers):
    if i<2:
        print(answer[0], answer[1], answer[2], answer[3], answer[4])

with open("data/whole_data_bowon.tsv", "w", encoding="utf-8") as f:
    field_names = ["original_text", "morphed_text", "unk_tokens", "not_unk_tokens", "entity"]
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(field_names)
    for answer in answers:
        writer.writerow([answer[0], answer[1], answer[2], answer[3], answer[4]])
