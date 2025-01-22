from os import path
import json
import tensorflow as tf, pandas as pd
import tensorflow_text  # Required to run exported model.
import sys
from tqdm import tqdm

DATASET_BUCKET = "/mnt/personal/ullriher/models/tf/decontext_dataset"
SAVED_MODEL_PATH = path.join("/mnt/personal/ullriher/models/tf/decontext_dataset", "t5_base/1611267950")

# arse first argument as model name
model_name = sys.argv[1]
metric = "decontextualization"
print(f"Computing metric: {metric}")
print(f"Using model {model_name}")

def load_predict_fn(model_path):
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures["serving_default"](tf.constant(x))["outputs"].numpy()

predict_fn = load_predict_fn(SAVED_MODEL_PATH)

def decontextualize(input):
    return predict_fn([input])[0].decode("utf-8")

dp = {}
def decontextualize_with_dp(input):
    if input in dp:
        return dp[input]
    result = decontextualize(input)
    dp[input] = result
    return result

def create_input(paragraph, target, page_title="", section_title=""):
    prefix = paragraph
    return " [SEP] ".join((page_title, section_title, prefix, target, ""))

def same_alphabetic_chars(generated, decontext_proposed):
    a = "".join(filter(str.isalpha, generated)).lower()
    b = "".join(filter(str.isalpha, decontext_proposed)).lower()
    return  bool(len(a)) and a == b
    

df = pd.read_json("/mnt/data/factcheck/claim_extraction/feversum/hf_multiclaim/test.jsonl", lines=True)
df["generated"]=None
predictions = f"/home/ullriher/ullriher/data/_paper/predictions/{model_name}.jsonl"
outfile = f"/home/ullriher/ullriher/data/_paper/metrics/{metric}/{model_name}.jsonl"
skip_title_from_context = True

with open(predictions, "r") as f:
    for i, line in enumerate(f):
        df.at[i, "generated"] = json.loads(line)
        # if context starts with source\n, remove it
        if skip_title_from_context:
            if df.at[i, "sentence_context"].startswith(df.at[i, "source"]+"\n"):
                df.at[i, "sentence_context"] = df.at[i, "sentence_context"][len(df.at[i, "source"])+1:]

# expand df by generated, one row per element of generated list
df = df.explode("generated").reset_index(drop=True)
df.drop(columns=["source_text"], inplace=True)
# remove leading source\n from sentence_context
df["decontext_result"] = None
df["decontext_label"] = None
df["decontext_proposed"] = None

if path.exists(outfile):
    df = pd.read_json(outfile, lines=True)
    print(f"Loaded checkpoint from {outfile}")

# if claims column, drop it
if "claims" in df.columns:
    df.drop(columns=["claims"], inplace=True)

df = df.dropna(subset=["generated"])
    
print("predicting")
for index, row in tqdm(df.iterrows()):
    if row["decontext_label"] is not None:
        continue
    
    page_title = row["source"]
    section_title = ""
    input = create_input(row["sentence_context"], row["generated"], page_title, section_title)
    decontextualized = decontextualize(input)
    
    try:
        row["decontext_label"], row["decontext_proposed"] = [s.strip() for s in decontextualized.split("####", 1)]
    except:
        row["decontext_label"] = ""
        row["decontext_proposed"] = ""
    
    if same_alphabetic_chars(row["generated"], row["decontext_proposed"]):
        row["decontext_label"] = "UNNECESSARY"
        
    # only preserve alphabetic and turn to uppercase
    df.at[index, "decontext_result"] = decontextualized
    df.at[index, "decontext_label"] = "".join(filter(str.isalpha, row["decontext_label"])).upper()
    df.at[index, "decontext_proposed"] = row["decontext_proposed"]
    # break at 100
    # save df to outfile
    if True or index % 100 == 0:
        df.to_json(outfile, lines=True, orient="records")