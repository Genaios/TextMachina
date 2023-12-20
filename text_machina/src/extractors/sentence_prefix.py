import re
from random import randint
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class SentencePrefix(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)

        regex = r"{(sentences(@\d+?)?)}"
        self.placeholder, n_sentences_match = re.findall(
            regex, self.input_config.template
        )[0]

        # Random if not sentence length specified,
        # or sentence length if specified.
        if not n_sentences_match:
            self.n_sentences = lambda x: randint(1, max(1, len(x) - 1))
        else:
            self.n_sentences = lambda x: int(n_sentences_match.replace("@", ""))

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

        return {self.placeholder: output_texts}
