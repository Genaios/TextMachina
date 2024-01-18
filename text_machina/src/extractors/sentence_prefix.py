from random import randint
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class SentencePrefix(Extractor):
    """
    Extractor that fills the prompt template with a sentence
    prefix extracted from a text column of a dataset.

    This extractor needs a template placeholder named {sentences}.

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - k (int): number of sentences in the prefix. If not specified,
                   `k` will be random for each sample.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("sentence_prefix", {})
        self.n_sentences = lambda x: self.args.get(
            "k", randint(1, max(1, len(x) - 1))
        )
        self.sampled_positions: List[int] = []

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        """
        For detection and attribution tasks, removes the extracted prefix
        from human texts to ensure both generations and human texts are
        continuations of sentence prefixes.

        For boundary tasks (human followed by generated), returns the prefix.

        Args:
            human_texts (List[str]): list of human texts.

        Returns:
            List[str]: prepared human texts.
        """
        docs = spacy_pipeline(
            human_texts,
            language=self.input_config.language,
            disable_pipes=[
                "ner",
                "tagger",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        output: List[str] = []
        for idx, doc in enumerate(docs):
            doc_sents = list(doc.sents)
            n_sentences = self.sampled_positions[idx]
            if self.task_type in [TaskType.DETECTION, TaskType.ATTRIBUTION]:
                text = "".join(
                    sent.text_with_ws for sent in doc_sents[n_sentences:]
                )
            else:
                text = "".join(
                    sent.text_with_ws for sent in doc_sents[:n_sentences]
                )
            output.append(text)
        return output

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        docs = spacy_pipeline(
            dataset[self.input_config.dataset_text_column],
            language=self.input_config.language,
            disable_pipes=[
                "ner",
                "tagger",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        output_texts = []
        for doc in docs:
            doc_sents = list(doc.sents)
            n_sentences = self.n_sentences(doc_sents)
            self.sampled_positions.append(n_sentences)
            output_texts.append(
                "".join([sent.text_with_ws for sent in doc_sents[:n_sentences]])
            )

        return {"sentences": output_texts}
