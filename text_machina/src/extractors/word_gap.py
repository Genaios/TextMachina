import re
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class WordGap(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        regex = r"{(.*(@\d+?)?)}"
        self.placeholder, self.n_words = re.findall(
            regex, self.input_config.template
        )[0]
        print(self.n_words)
        exit()

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        """
        For detection and attribution tasks, removes the extracted prefix
        from human texts to ensure both generations and human texts are
        continuations of word prefixes.

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
            n_words = self.sampled_positions[idx]
            if self.task_type in [TaskType.DETECTION, TaskType.ATTRIBUTION]:
                text = "".join(token.text_with_ws for token in doc[n_words:])
            else:
                text = "".join(token.text_with_ws for token in doc[:n_words])
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
            n_words = self.n_words(doc)
            self.sampled_positions.append(n_words)
            output_texts.append(
                "".join([token.text_with_ws for token in doc[:n_words]])
            )

        return {self.placeholder: output_texts}
