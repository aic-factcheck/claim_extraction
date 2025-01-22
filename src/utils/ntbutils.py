import pandas as pd
import os
import torch
from utils.dbutils import *
from utils.datautils import *
from collections import OrderedDict
import pickle
import sys
from huggingface_hub import HfApi


def hf_upload(dirs=[], repo_id="", repo_type="model", private=False):
    if isinstance(dirs, str):
        dirs = [dirs]
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    except:
        pass

    for dir in dirs:
        api.upload_folder(repo_id=repo_id, folder_path=dir, repo_type=repo_type)


def pdump(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def pload(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pd_setup(colwidth=100, precision=2):
    pd.set_option("display.max_colwidth", colwidth)
    pd.set_option("display.precision", precision)


def render(dataframe):
    """Display a dataframe with vertical column headers"""
    styles = [
        dict(selector="th", props=[("width", "40px")]),
        dict(
            selector="th.col_heading",
            props=[
                ("text-align", "left"),
                ("writing-mode", "vertical-rl"),
                ("transform", "rotateZ(180deg)"),
                ("height", "290px"),
                ("vertical-align", "bottom"),
            ],
        ),
    ]
    return dataframe.fillna("").style.set_table_styles(styles).background_gradient(cmap="Blues")


def is_cuda():
    return bool(torch.cuda.device_count())


def scoreboard(df):
    return df.mean(axis=0).sort_values(ascending=False)


def scoredict(df):
    return scoreboard(df).to_dict(OrderedDict)


def load_user_libs(lib_root="/home/ullriher/lib", must_contain=".path_include"):
    # export PATH=$(find ~/lib -type f -name '*.path_include' | sed -r 's|/[^/]+$||' |sort |uniq | paste -sd ":" -):$PATH
    lib_root = "/home/ullriher/lib"
    libs = []
    for dirpath, dirnames, filenames in os.walk(lib_root):
        if must_contain and must_contain in filenames:
            libs.append(dirpath)
            sys.path.insert(0, dirpath)


DEVICE = "cuda" if is_cuda() else "cpu"
