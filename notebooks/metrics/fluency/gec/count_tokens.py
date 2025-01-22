from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

#load GPT2 dataset
dataset_owt = load_dataset("Skylion007/openwebtext")

#load GPT2 tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Access vocabulary from tokenizer
vocab = tokenizer.get_vocab()

#init token counts
token_counts = {}

#iterate through texts and count
for idx in tqdm(range(len(dataset_owt['train']))):

    tokenized_text = tokenizer(dataset_owt['train'][idx]['text'], return_tensors='pt')

    # Convert IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"].flatten())

    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

#save as JSON
try:
    import json
    with open("/home/mlynatom/data/gpt2_counts.json", "w") as outfile: 
        json.dump(token_counts, outfile)
except:
    print("Error while saving JSON file.")

#save to pickle (redundancy)
try:
    import pickle
    with open("/home/mlynatom/data/gpt2_counts.pickle", "wb") as outfile:
        pickle.dump(token_counts, outfile, pickle.HIGHEST_PROTOCOL)
except:
    print("Error while saving pickle!")
