from os import path
import json
import bert_score
import sys
from tqdm import tqdm
from utils.datautils import extract_triplets
from transformers import pipeline
from utils.ntbutils import load_user_libs
import pysbd
from utils.datautils import avg_top_n
import pandas as pd

load_user_libs("/home/ullriher/lib", ".path_include")
from alignscore import AlignScore

#segmenter 
segmenter = pysbd.Segmenter(language="en", clean=False)
sent_tokenize = segmenter.segment

align_large = AlignScore(model='roberta-large', batch_size=32, device="cuda:0", ckpt_path='/home/ullriher/ullriher/models/alignscore/AlignScore-large.ckpt', evaluation_mode='nli_sp')
align_base = AlignScore(model='roberta-base', batch_size=32, device="cuda:0", ckpt_path='/home/ullriher/ullriher/models/alignscore/AlignScore-base.ckpt', evaluation_mode='nli_sp')


def bertscore_solve(premise, claim):
    text_sentencewise = sent_tokenize(premise)
    P, R, F = bert_score.score([claim] * len(text_sentencewise), text_sentencewise, model_type="roberta-base")

    return (float(avg_top_n(P, 2)),float(avg_top_n(R, 2)),float(avg_top_n(F, 2)))


# arse first argument as model name
if True:
    model_name = sys.argv[1]
else:
    model_name = "t5_small_multiclaim"

metric = "faithfulness"

print(f"Computing metric: {metric}")
print(f"Using model {model_name}")


df = pd.read_json("/mnt/data/factcheck/claim_extraction/feversum/hf_multiclaim/test.jsonl", lines=True)
df["generated"] = None
predictions = f"/home/ullriher/ullriher/data/_paper/predictions/{model_name}.jsonl"
outfile = f"/home/ullriher/ullriher/data/_paper/metrics/{metric}/{model_name}.jsonl"
skip_title_from_context = False

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
df["bertscore"] = None
df["bertscore_avgtop2"] = None
df["alignscore_base"] = None
df["alignscore_large"] = None

if path.exists(outfile):
    df = pd.read_json(outfile, lines=True)
    print(f"Loaded checkpoint from {outfile}")

# if claims column, drop it
if "claims" in df.columns:
    df.drop(columns=["claims"], inplace=True)

# if claims column, drop it
if "claims" in df.columns:
    df.drop(columns=["claims"], inplace=True)

df = df.dropna(subset=["generated"])
print("predicting")
for index, row in tqdm(df.iterrows()):
    if row["alignscore_large"] is not None:
        continue

    premise = row["sentence_context"]
    claim = row["generated"]
    
    df.at[index, "bertscore"] = tuple(map(float, bert_score.score([premise], [claim], model_type="roberta-base")))
    df.at[index, "bertscore_avgtop2"] = bertscore_solve(premise, claim)
    df.at[index, "alignscore_base"] = align_base.score(contexts=[premise], claims=[claim])[0]
    df.at[index, "alignscore_large"] = align_large.score(contexts=[premise], claims=[claim])[0]
    
    # break at 100
    # save df to outfile
    if True or index % 100 == 0:
        df.to_json(outfile, lines=True, orient="records")