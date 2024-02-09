from math import ceil
from random import randint, sample, uniform
from typing import Any, Dict, List

from datasets import Dataset

from ..common.exceptions import ExtractorInvalidArgs
from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class SentenceRewriting(Extractor):
    """
    Extractor that fills the prompt template with a sentence that
    has to be rewritten by an LLM.

    This extractor needs two template placeholders:


    This extractor needs two template placeholders:
        - {sentence}: will be filled with sentence to be rewritten.

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - percentage_range (List[float]): range delimiting the percentage
            of sentences to be rewritten. At least one sentence will be
            always rewritten.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        args: Dict[str, Any] = input_config.extractor_args.get(
            "sentence_rewriting", {}
        )
        workspace: Dict[str, Any] = {"positions": [], "human_spans": []}
        super().__init__(input_config, task_type, workspace, args)

    def check_valid_args(self):
        mandatory_args = ["percentage_range"]
        for mandatory_arg in mandatory_args:
            if mandatory_arg not in self.args:
                raise ExtractorInvalidArgs(
                    self.__class__.__name__, mandatory_args
                )

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return [
            "".join(doc_sentences)
            for doc_sentences in self.workspace["human_spans"]
        ]

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        text_column = self.input_config.dataset_text_column
        texts = spacy_pipeline(
            texts=dataset[text_column], language=self.input_config.language
        )
        sentences_to_rewrite = []
        for text in texts:
            sentences = list([sent.text_with_ws for sent in text.sents])
            # Skip the sample if there are no sentences.
            if len(sentences) == 0:
                continue
            self.workspace["human_spans"].append(sentences)
            n_sentences_to_select = randint(
                1,
                ceil(uniform(*self.args["percentage_range"]) * len(sentences)),
            )
            sampled_positions = sorted(
                sample(list(range(len(sentences))), n_sentences_to_select)
            )
            self.workspace["positions"].append(sampled_positions)
            for position in sampled_positions:
                sentences_to_rewrite.append(sentences[position])
        return {
            "sentence": list(map(str, sentences_to_rewrite)),
        }
