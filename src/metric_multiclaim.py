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
import numpy as np

load_user_libs("/home/ullriher/lib", ".path_include")
from alignscore import AlignScore
from factsumm import FactSumm
from sentence_transformers import CrossEncoder

deberta = CrossEncoder("cross-encoder/nli-deberta-v3-small")

# segmenter
# factsumm = FactSumm()
segmenter = pysbd.Segmenter(language="en", clean=False)
sent_tokenize = segmenter.segment

# align_large = AlignScore(model='roberta-large', batch_size=32, device="cuda:0", ckpt_path='/home/ullriher/ullriher/models/alignscore/AlignScore-large.ckpt', evaluation_mode='nli_sp', verbose=False)
align_base = AlignScore(
    model="roberta-base",
    batch_size=32,
    device="cuda:0",
    ckpt_path="/home/ullriher/ullriher/models/alignscore/AlignScore-base.ckpt",
    evaluation_mode="nli_sp",
    verbose=False,
)


def focus(gold_claims, predicted_claims, same=False):
    if not isinstance(gold_claims, list):
        gold_claims = gold_claims.split("\n")
    if not isinstance(predicted_claims, list):
        predicted_claims = predicted_claims.split("\n")
    try:

        result = []
        for claim in predicted_claims:
            gold_claims_copy = gold_claims.copy()
            if same:  # pop claim from gold_pairs
                # copy gold claims
                for j in range(len(gold_claims_copy)):
                    if claim == gold_claims_copy[j]:
                        gold_claims_copy.pop(j)
                        break
            scores = deberta.predict(
                list(zip(gold_claims_copy, [claim] * len(gold_claims_copy))),
                apply_softmax=True,
                show_progress_bar=False,
            )[:, 1]
            result.append(np.max(scores))
        return result, np.mean(result)
    except Exception as e:
        print(e)
        return [], np.nan


def focus_alignscore(gold_claims, predicted_claims, align=align_base, same=False):
    try:
        if not isinstance(gold_claims, list):
            gold_claims = gold_claims.split("\n")
        if not isinstance(predicted_claims, list):
            predicted_claims = predicted_claims.split("\n")
        if len(gold_claims) == 0 or len(predicted_claims) == 0:
            return [], 0
        align.verbose = False
        result = []
        for claim in predicted_claims:
            gold_claims_copy = gold_claims.copy()
            if same:  # pop claim from gold_pairs
                # copy gold claims
                for j in range(len(gold_claims_copy)):
                    if claim == gold_claims_copy[j]:
                        gold_claims_copy.pop(j)
                        break
            score = align.score(["\n".join(gold_claims_copy)], [claim])[0]
            result.append(score)
        return result, np.mean(result)
    except Exception as e:
        print(e)
        return [], np.nan


# arse first argument as model name
if True:
    model_name = sys.argv[1]
else:
    model_name = "t5_small_multiclaim"

metric = "multiclaim"

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
# df = df.explode("generated").reset_index(drop=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=["source_text"], inplace=True)
# remove leading source\n from sentence_context
df["claims"] = df["claims"].str.split("\n")

df["f_deberta"] = None
df["c_deberta"] = None
df["r_deberta"] = None
df["f_align"] = None
df["c_align"] = None
df["r_align"] = None
df["f_align_mean"] = None
df["f_deberta_mean"] = None
df["c_deberta_mean"] = None
df["c_align_mean"] = None
df["r_deberta_mean"] = None
df["r_align_mean"] = None

if path.exists(outfile):
    df = pd.read_json(outfile, lines=True)
    print(f"Loaded checkpoint from {outfile}")

df = df.dropna(subset=["generated"])
print("predicting")


for index, row in tqdm(df.iterrows()):
    if row["r_align"] is not None:
        continue
    claims = list(row["generated"].copy())
    gold_claims = list(row["claims"].copy())

    # if claims or generated is empty, skip
    df.at[index, "f_deberta"], df.at[index, "f_deberta_mean"] = focus(gold_claims, claims)
    df.at[index, "c_deberta"], df.at[index, "c_deberta_mean"] = focus(claims, gold_claims)
    df.at[index, "r_deberta"], df.at[index, "r_deberta_mean"] = focus(claims, claims, same=True)
    df.at[index, "f_align"], df.at[index, "f_align_mean"] = focus_alignscore(gold_claims, claims)
    df.at[index, "c_align"], df.at[index, "c_align_mean"] = focus_alignscore(claims, gold_claims)
    df.at[index, "r_align"], df.at[index, "r_align_mean"] = focus_alignscore(claims, claims, same=True)

    # break at 100
    # save df to outfile
    if True or index % 100 == 0:
        df.to_json(outfile, lines=True, orient="records")
