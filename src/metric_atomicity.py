from os import path
import json
import tensorflow as tf, pandas as pd
import tensorflow_text  # Required to run exported model.
import sys
from tqdm import tqdm
from utils.datautils import extract_triplets
from transformers import pipeline
from utils.ntbutils import load_user_libs

load_user_libs("/home/ullriher/lib", ".path_include")
from factsumm import FactSumm


def count_facts(facts, floor=1):
    symmetries = {frozenset((fact[0], fact[2])) for fact in facts}
    return max(len(symmetries), floor)


rebel = pipeline("text2text-generation", model="Babelscape/rebel-large", tokenizer="Babelscape/rebel-large")


def rebel_solve(input_text):
    preds = rebel(input_text, return_tensors=True, return_text=False)
    out = rebel.tokenizer.batch_decode([preds[0]["generated_token_ids"]])
    return extract_triplets(out[0])


factsumm = FactSumm()


def factsumm_solve(claim):
    if isinstance(factsumm.ner, str) and isinstance(factsumm.rel, str):
        factsumm.extract_facts("", "", device="cuda:0")
    claim_lines = factsumm._segment(claim)
    claim_ents = tuple(factsumm.ner(claim_lines))
    claim_facts = factsumm.get_facts(claim_lines, claim_ents)

    return claim_facts


# arse first argument as model name
if True:
    model_name = sys.argv[1]
else:
    model_name = "t5_small_multiclaim"

metric = "atomicity"

print(f"Computing metric: {metric}")
print(f"Using model {model_name}")


df = pd.read_json("/mnt/data/factcheck/claim_extraction/feversum/hf_multiclaim/test.jsonl", lines=True)
df["generated"] = None
predictions = f"/home/ullriher/ullriher/data/_paper/predictions/{model_name}.jsonl"
outfile = f"/home/ullriher/ullriher/data/_paper/metrics/{metric}/{model_name}.jsonl"
skip_title_from_context = True

with open(predictions, "r") as f:
    for i, line in enumerate(f):
        df.at[i, "generated"] = json.loads(line)
        # if context starts with source\n, remove it
        if skip_title_from_context:
            if df.at[i, "sentence_context"].startswith(df.at[i, "source"] + "\n"):
                df.at[i, "sentence_context"] = df.at[i, "sentence_context"][len(df.at[i, "source"]) + 1 :]

# expand df by generated, one row per element of generated list
df = df.explode("generated").reset_index(drop=True)
df.drop(columns=["source_text"], inplace=True)
# remove leading source\n from sentence_context
df["rebel"] = None
df["factsumm"] = None
df["rebel_facts"] = None
df["factsumm_facts"] = None

if path.exists(outfile):
    df = pd.read_json(outfile, lines=True)
    print(f"Loaded checkpoint from {outfile}")

columns_to_drop = ["claims", "sentence", "sentence_context"]
for column in columns_to_drop:
    if column in df.columns:
        df.drop(columns=[column], inplace=True)

df = df.dropna(subset=["generated"])

print("predicting")
for index, row in tqdm(df.iterrows()):
    if row["factsumm_facts"] is not None:
        continue


    # batch

    # only preserve alphabetic and turn to uppercase
    df.at[index, "rebel_facts"] = rebel_solve(row["generated"])
    df.at[index, "factsumm_facts"] = factsumm_solve(row["generated"])
    df.at[index, "rebel"] = count_facts(df.at[index, "rebel_facts"])
    df.at[index, "factsumm"] = count_facts(df.at[index, "factsumm_facts"])
    
    # break at 100
    # save df to outfile
    if True or index % 100 == 0:
        df.to_json(outfile, lines=True, orient="records")