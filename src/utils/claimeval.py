from factsumm.factsumm import FactSumm
import os
from os import path

import tensorflow as tf
import tensorflow_text  # Required to run exported model.

DECONTEXT_PATH = "/mnt/personal/ullriher/models/tf/decontext_dataset"
DECONTEXT_MODELS = {
    "base": f"{DECONTEXT_PATH}/t5_base/1611267950",
    "3B": f"{DECONTEXT_PATH}/t5_3B/1611333896",
    "11B": f"{DECONTEXT_PATH}/t5_11B/1605298402",
}


class ClaimEvaluation:
    """Class for claim evaluation"""

    def __init__(
        self, factsumm=None, sentence_segmentation=None, bert_score_model=None, decontextualization_size="11B"
    ):
        """Initialize the class
        Args:
            factsumm (FactSumm): FactSumm object
            sentence_segmentation (function): function for sentence segmentation
            bert_score_model (function): function for bert score
        """

        if factsumm is None:
            factsumm = FactSumm()
        if sentence_segmentation is None:
            sentence_segmentation = factsumm._segment
        if bert_score_model is None:
            bert_score_model = factsumm.bert_score
        

        self.factsumm = factsumm
        self.sentence_segmentation = sentence_segmentation
        self.bert_score_model = bert_score_model
        print("ClaimEvaluation initialized")
