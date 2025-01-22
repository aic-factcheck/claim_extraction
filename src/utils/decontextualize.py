import os
from os import path

import tensorflow as tf
import tensorflow_text  # Required to run exported model.

MODEL_SIZE = "base"  # @param["base", "3B", "11B"]

DATASET_BUCKET = "/mnt/personal/ullriher/models/tf/decontext_dataset"

SAVED_MODELS = {
    "base": f"{DATASET_BUCKET}/t5_base/1611267950",
    "3B": f"{DATASET_BUCKET}/t5_3B/1611333896",
    "11B": f"{DATASET_BUCKET}/t5_11B/1605298402",
}

SAVED_MODEL_PATH = SAVED_MODELS[MODEL_SIZE]
DEV = path.join(DATASET_BUCKET, "decontext_dev.jsonl")
SAVED_MODEL_PATH = path.join(DATASET_BUCKET, "t5_base/1611267950")


def load_predict_fn(model_path):
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures["serving_default"](tf.constant(x))["outputs"].numpy()


predict_fn = load_predict_fn(SAVED_MODEL_PATH)


def decontextualize(claim, prefix="", suffix="", page_title="", section_title=""):
    input = " [SEP] ".join(
        (
            page_title,
            section_title,
            prefix,
            claim,
            suffix
        )
    )
    return predict_fn([input])[0].decode("utf-8")


def create_input(paragraph, target_sentence_idx, page_title="", section_title=""):
    """Creates a single Decontextualization example input for T5.

    Args:
        paragraph: List of strings. Each string is a single sentence.
        target_sentence_idx: Integer index into `paragraph` indicating which
            sentence should be decontextualized.
        page_title: Optional title string. Usually Wikipedia page title.
        section_title: Optional title of section within page.
    """
    prefix = " ".join(paragraph[:target_sentence_idx])
    target = paragraph[target_sentence_idx]
    suffix = " ".join(paragraph[target_sentence_idx + 1 :])
    return " [SEP] ".join((page_title, section_title, prefix, target, suffix))
