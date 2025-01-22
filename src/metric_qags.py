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


# segmenter
# factsumm = FactSumm()
segmenter = pysbd.Segmenter(language="en", clean=False)
sent_tokenize = segmenter.segment

from factsumm import FactSumm
from factsumm.utils.utils import qags_score

factsumm = FactSumm()


def focus(gold_claims: list[str], pred_claims: list[str], verbose: bool = False, same=False) -> float:
    try:
        if isinstance(factsumm.qg, str) or isinstance(factsumm.qa, str) or isinstance(factsumm.ner, str):
            factsumm.extract_qas("b", " ".join(pred_claims), verbose=False, device="cuda:0")

        # gold_entities = factsumm.ner(gold_claims)
        pred_entities = factsumm.ner(pred_claims)
        Q = factsumm.qg(pred_claims, pred_entities)

        gold_answers = factsumm.qa(" ".join(gold_claims), Q)
        pred_answers = factsumm.qa(" ".join(pred_claims), Q)

        if verbose:
            factsumm._print_qas("gold", gold_answers)
            factsumm._print_qas("pred", pred_answers)

        focus = qags_score(gold_answers, pred_answers)
        if verbose:
            print(f"QAGS Score: {focus}\n")

        return focus, pred_entities, Q, gold_answers, pred_answers
    except Exception as e:
        print(e)
        return np.nan, [], [], [], []


# arse first argument as model name
if True:
    model_name = sys.argv[1]
else:
    model_name = "t5_small_multiclaim"

metric = "qags"

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

df["f"] = None
df["c"] = None
df["f_q"] = None
df["f_gold_a"] = None
df["f_a"] = None
df["f_ents"] = None
df["c_q"] = None
df["c_gold_a"] = None
df["c_a"] = None
df["c_ents"] = None

if path.exists(outfile):
    df = pd.read_json(outfile, lines=True)
    print(f"Loaded checkpoint from {outfile}")

df = df.dropna(subset=["generated"])
print("predicting")


for index, row in tqdm(df.iterrows()):
    if row["c_ents"] is not None:
        continue
    claims = list(row["generated"].copy())
    gold_claims = list(row["claims"].copy())

    # if claims or generated is empty, skip
    f = focus(gold_claims, claims)
    c = focus(claims, gold_claims)

    df.at[index, "f"] = f[0]
    df.at[index, "f_q"] = f[2]
    df.at[index, "f_gold_a"] = f[3]
    df.at[index, "f_a"] = f[4]
    df.at[index, "f_ents"] = f[1]
    df.at[index, "c"] = c[0]
    df.at[index, "c_q"] = c[2]
    df.at[index, "c_gold_a"] = c[3]
    df.at[index, "c_a"] = c[4]
    df.at[index, "c_ents"] = c[1]

    # break at 100
    # save df to outfile
    if True or index % 100 == 0:
        df.to_json(outfile, lines=True, orient="records")
