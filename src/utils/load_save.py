import json
from typing import Dict, List

import pandas as pd


def save_listdict2jsonl(path: str, data: List[Dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(data_point, ensure_ascii=False,
                     sort_keys=True) + "\n" for data_point in data)


def save_dict2json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, sort_keys=False))


def load_csv_dataset2dict(csv_file_path: str, separator: str = ";"):
    dataframe = pd.read_csv(csv_file_path, sep=separator)

    return dataframe.to_dict("records")


def load_jsonl2list(path: str) -> List[Dict]:
    """
    Load dataset from path in json format into list of python dictionaries.
    """
    data = []

    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data
