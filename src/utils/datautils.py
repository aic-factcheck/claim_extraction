import json
import os
import pickle
import random
import statistics
import unicodedata as ud
import urllib.parse
import urllib.request
from datetime import datetime

from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split

#  The id's that were used in the first three "hacky" iterations of this scheme
LABEL_NUM = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
LABEL_STR = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}


def detokenize2(txt):
    # updated detokenize, most models are not trained with this...
    txt = (
        txt.replace(" .", ".")
        .replace("-COLON-", ":")
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace(" :", ":")
        .replace(" ;", ";")
    )
    txt = (
        txt.replace("`` ", '"')
        .replace(" ''", '"')
        .replace(" '", "'")
        .replace("--", "-")
    )
    txt = txt.replace("-LRB- ", "(").replace("-LRB-", "(").replace("-RRB-", ")")
    txt = txt.replace("-LSB- ", "[").replace("-LSB-", "[").replace("-RSB-", "]")
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace("[ ", "[").replace(" ]", "]")
    return txt


def load_api_export(
    format="nli",
    evidence_format="text",
    simulate_nei_evidence=1,
    single_evidence=0,
    include_twitter=0,
):
    params = {
        "format": format,
        "evidenceFormat": evidence_format,
        "simulateNeiEvidence": simulate_nei_evidence,
        "singleEvidence": single_evidence,
        "includeTwitter": include_twitter,
    }
    result = []
    with urllib.request.urlopen(
        "https://fcheck.fel.cvut.cz/label/export?" + urllib.parse.urlencode(params)
    ) as url:
        for line in url:
            result.append(json.loads(line.decode("utf-8")))
    return result


def load_jsonl(location="../export-snapshots/export_08-27-2021_0655pm_173.jsonl"):
    with open(location, "r", encoding="utf-8") as file:
        result = []
        for line in file:
            result.append(json.loads(line))
    return result


def save_splits(train, validation, test, folder=None):
    if folder is None:
        folder = f"../data/dump{datetime.now().strftime('_%d-%m-%Y_%H-%M-%S')}"
    if not os.path.exists(folder):
        os.mkdir(folder)

    save_jsonl(train, folder + "/train.jsonl")
    save_jsonl(validation, folder + "/validation.jsonl")
    save_jsonl(test, folder + "/test.jsonl")


def save_jsonl(data, location=None):
    if location is None:
        location = f"../data/dump{datetime.now().strftime('_%d-%m-%Y_%H-%M-%S')}.jsonl"
    with open(location, "w", encoding="utf-8") as f:
        for datapoint in data:
            print(json.dumps(datapoint, ensure_ascii=False), file=f)


def expand_by_evidence(dataset):
    result, label_id = [], 1
    for datapoint in dataset:
        for evidence in datapoint["evidence"]:
            datapoint_expanded = datapoint.copy()
            datapoint_expanded["label_id"], datapoint_expanded["evidence"] = (
                label_id,
                evidence,
            )
            result.append(datapoint_expanded)
        label_id += 1
    return result


def collapse_by(dataset, parameter="source"):
    result = {}
    for datapoint in dataset:
        if datapoint[parameter] not in result:
            result[datapoint[parameter]] = []
        result[datapoint[parameter]].append(datapoint)
    result = list(result.values())
    labels = []
    for collapsed in result:
        s = r = n = t = 0
        for datapoint in collapsed:
            s += datapoint["label"] == "SUPPORTS"
            r += datapoint["label"] == "REFUTES"
            n += datapoint["label"] == "NOT ENOUGH INFO"
            t += 1
        labels += [
            "NOT ENOUGH INFO"
            if max(s, r, n) == n
            else ("REFUTES" if max(s, r, n) == r else "SUPPORTS")
        ]
    return result, labels


def counter(dataset):
    if isinstance(dataset[0], InputExample):
        dataset = [LABEL_STR[datapoint.label] for datapoint in dataset]
    elif "label" in dataset[0]:
        dataset = [datapoint["label"] for datapoint in dataset]
    values = sorted(list(set(dataset)))
    return [
        (value, dataset.count(value), dataset.count(value) / len(dataset))
        for value in values
    ]


def expand_collapsed(dataset):
    result = []
    for datapoints in dataset:
        result.extend(datapoints)
    return result


def split(
    dataset,
    test_size=0.12,
    validation_size=0.12,
    skip_ids=tuple(),
    leakage_prevention_level="source",
    seed=1234,
):
    # TODO: uniform val-test split
    skipped = [datapoint for datapoint in dataset if datapoint["id"] in skip_ids]
    dataset = [datapoint for datapoint in dataset if datapoint["id"] not in skip_ids]

    dataset, labels = collapse_by(dataset, leakage_prevention_level)

    if isinstance(test_size, float):
        test_size = int(test_size * len(dataset))
    if isinstance(validation_size, float):
        validation_size = int(validation_size * len(dataset))

    train, test, train_labels, test_labels = train_test_split(
        dataset, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train, validation, train_labels, validation_labels = train_test_split(
        train,
        train_labels,
        test_size=validation_size,
        random_state=seed,
        stratify=train_labels,
    )
    train, validation, test = (expand_collapsed(d) for d in (train, validation, test))
    train.extend(skipped)
    random.Random(seed).shuffle(train)
    return train, validation, test


def to_examples(dataset, norm="NFC"):
    return [
        InputExample(
            datapoint["id"],
            [
                ud.normalize(norm, detokenize2(" ".join(datapoint["evidence"]))),
                ud.normalize(norm, datapoint["claim"]),
            ],
            LABEL_NUM[datapoint["label"]],
        )
        for datapoint in dataset
    ]


def load_examples_from_pickle(folder):
    with open(folder + "/trn_examples.p", "rb") as trn, open(
        folder + "/val_examples.p", "rb"
    ) as val, open(folder + "/tst_examples.p", "rb") as tst:
        return pickle.load(trn), pickle.load(val), pickle.load(tst)


#  location = "../export-snapshots/export_08-30-2021_0200am_173.jsonl"
#  dataset = load_jsonl(location)
#
#  dataset = expand_by_evidence(dataset)
#  train, validation, test = split(dataset)
#  print(counter(train), "\n", counter(validation), "\n", counter(test))

def avg(iterable):
    return sum(iterable)/len(iterable)

def avg_top_n(iterable, n):
    return avg(sorted(iterable)[-n:])

def hmean(data):
    try:
        return statistics.harmonic_mean(data)
    except:
        return 0

def extract_triplets(text):
    triplets = []
    subject, object_, relation = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets
