import transformers, os, getpass, shutil, yaml
from transformers import (
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
)
from huggingface_hub import Repository, create_repo
from urllib.request import urlopen

__version__ = "0.0.2"

USER = getpass.getuser()
DEFAULT_DIR = "/mnt/data/factcheck/models/" + USER + "_EXP"

# TODO: Install working git & git lfs! :-) Simple (not ultra safe) workaround:
# $ sh /mnt/data/factcheck/git/install.sh

FOOTER_TEMPLATE = """
{additional_readme}

## 🌳 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 👬 Authors
The model was trained and uploaded by **[{author}](https://udb.fel.cvut.cz/?uid={author}&sn=&givenname=&_cmd=Hledat&_reqn=1&_type=user&setlang=en)** (e-mail: [{author}@fel.cvut.cz](mailto:{author}@fel.cvut.cz))

The code was codeveloped by the NLP team at Artificial Intelligence Center of CTU in Prague ([AIC](https://www.aic.fel.cvut.cz/)).

## 🔐 License
[{license}](https://choosealicense.com/licenses/{license})
"""

DEFAULT_README_NLI = (
    """
# 🦾 {model_name}
Transformer model for **Natural Language Inference** in {languages} languages finetuned on {datasets} datasets.

## 🧰 Usage

### 👾 Using UKPLab `sentence_transformers` `CrossEncoder`
The model was trained using the `CrossEncoder` API and we recommend it for its usage.
```python
from sentence_transformers.cross_encoder import CrossEncoder
model = CrossEncoder('{target}')
scores = model.predict([["My first context.", "My first hypothesis."],  
                        ["Second context.", "Hypothesis."]])
```

### 🤗 Using Huggingface `transformers`
```python
from transformers import {model_classname}, {tokenizer_classname}
model = {model_classname}.from_pretrained("{target}")
tokenizer = {tokenizer_classname}.from_pretrained("{target}")
```

"""
    + FOOTER_TEMPLATE
)

DEFAULT_README_SUM = (
    """
# 🦾 {model_name}
Transformer model for **Claim Extraction** in {languages} languages finetuned using {datasets} datasets.

## 🧰 Usage

### 🤗 Using Huggingface `transformers`
```python
from transformers import {model_classname}, {tokenizer_classname}
model = {model_classname}.from_pretrained("{target}")
tokenizer = {tokenizer_classname}.from_pretrained("{target}")
```
"""
    + FOOTER_TEMPLATE
)

CITATION_TEMPLATE = """
## 💬 Citation
If you find this repository helpful, feel free to cite our publication:
```
{citation}
```
"""

NLI_TAGS = ["text-classification"]


def commit(local_dir, target, message):
    return Repository(local_dir=local_dir, clone_from=target, use_auth_token=True).commit(message)


def yaml_card(languages, tags, license, datasets=None):
    card = {
        "languages": languages,
        "tags": tags,
        "license": license,
    }
    if datasets is not None:
        card["datasets"] = datasets
    return ("---\n" + yaml.dump(card) + "\n---\n",)


def push_model_files(model_class, tokenizer_class, model_path, target, private=True, tf_model_class=None):
    model = model_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path)
    namespace, model_name = target.split("/")

    if namespace == "ctu-aic":  # TODO: remove - this is hotfix for Hf glitch
        create_repo(target, organization=namespace, private=private, exist_ok=True)
        model.push_to_hub(repo_path_or_name=target, organization=namespace)
        if tf_model_class is not None:
            tf_model_class.from_pretrained(model_path, from_pt=True).push_to_hub(
                repo_path_or_name=target, organization=namespace
            )
        tokenizer.push_to_hub(repo_path_or_name=target, organization=namespace)
    else:
        create_repo(target, private=private, exist_ok=True)
        model.push_to_hub(repo_path_or_name=target)
        tokenizer.push_to_hub(repo_path_or_name=target)
        if tf_model_class is not None:
            tf_model_class.from_pretrained(model_path, from_pt=True).push_to_hub(repo_path_or_name=target)


def upload_model(
    model_path,
    target,
    model_class=AutoModelForSequenceClassification,
    tf_model_class=TFAutoModelForSequenceClassification,
    tokenizer_class=AutoTokenizer,
    extra_files=[],
    local_copy=DEFAULT_DIR,
    languages=["cs"],
    tags=[],
    datasets=[],
    license="cc-by-sa-4.0",
    author=USER,
    readme=DEFAULT_README_NLI,
    additional_readme="",
    citation=None,
    private=True,
):
    if local_copy is not None:
        cwd = os.getcwd()
        os.chdir(local_copy)

    namespace, model_name = target.split("/")
    model_classname = model_class.__name__
    tokenizer_classname = tokenizer_class.__name__

    push_model_files(model_class, tokenizer_class, model_path, target, private, tf_model_class)
    if local_copy is not None:
        if readme is not None:
            with commit(model_name, target, "Generate README"):
                with open("README.md", "w") as freadme:
                    print(yaml_card(languages, tags, license, datasets), file=freadme)
                    print(readme.format(**locals()), file=freadme)
                    if citation is not None:
                        print(CITATION_TEMPLATE.format(**locals()), file=freadme)

        os.chdir(cwd)


def upload_dataset(
    dataset_path,
    target,
    extra_files=[],
    local_copy=DEFAULT_DIR,
    languages=["cs"],
    tags=[],
    license="cc-by-sa-4.0",
    author=USER,
    readme=DEFAULT_README_NLI,
    additional_readme="",
    citation=None,
    private=True,
):
    pass


CITATION_CTKFACTS = """
@article{DBLP:journals/corr/abs-2201-11115,
  author    = {Herbert Ullrich and
               Jan Drchal and
               Martin R{\'{y}}par and
               Hana Vincourov{\'{a}} and
               V{\'{a}}clav Moravec},
  title     = {CsFEVER and CTKFacts: Acquiring Czech Data for Fact Verification},
  journal   = {CoRR},
  volume    = {abs/2201.11115},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.11115},
  eprinttype = {arXiv},
  eprint    = {2201.11115},
  timestamp = {Tue, 01 Feb 2022 14:59:01 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-11115.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""
