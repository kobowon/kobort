import csv
from tokenizer import make_tokens
answers = []
with open("test.tsv", "r", encoding="utf-8") as f:
    lines = csv.reader(f, delimiter='\t',)
    for i, line in enumerate(lines):
        if i>0:
            #print(line)
            origin_text, _, _, label_str = line
            morphed_text = make_tokens(origin_text, model_name="mecab")
            tokenized_text = make_tokens(origin_text, model_name="wp-mecab")
            tokenized_text = ' '.join(tokenized_text)
            answers.append([origin_text, morphed_text, tokenized_text, label_str])

for i, answer in enumerate(answers):
    if i<2:
        print(answer[0], answer[1], answer[2], answer[3])

with open("test_v1.1.tsv", "w", encoding="utf-8") as f:
    field_names = ["original_text", "morphed_text", "tokenize", "bio_tagging"]
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(field_names)
    for answer in answers:
        writer.writerow([answer[0], answer[1], answer[2], answer[3]])

        