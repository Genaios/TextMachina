from math import ceil
from random import randint
from typing import Dict, List, Tuple

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class GapSentence(Extractor):
    """
    Extractor that fills the prompt template with a boundary of
    two sentences (left-side and right-side of a sampled sentence),
    and with the number of sentences the LLM has to generate in
    between the boundary sentences.

    This extractor needs two template placeholders:
        - {n}: will be filled with the number of sentences to generate
               between the boundary sentences.
        - {boundaries}: will be filled with the boundary sentences separated
                        by the gap token and newlines.
                        E.g., "sentence 1.\n____\nsentence2"

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - gap_token (str): gap token, e.g., "____"
        - max_percentage_boundaries (float): max percentage of
          boundaries to sample from a text. In a text of N sentences,
          there will be N-1 possible boundaries of two sentences.
        - max_sentence_span (int): max number of sentences to be generated
          between the boundary sentences.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("gap_sentence", {})
        self.workspace = {
            "positions": [],
            "human_spans": [],
            "num_boundaries": [],
        }

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return [
            " ".join(doc_sentences)
            for doc_sentences in self.workspace["human_spans"]
        ]

    def _format_boundary(self, pair: Tuple[str, str]) -> str:
        return f"{pair[0]}\n{self.args['gap_token']}\n{pair[1]}"

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        text_column = self.input_config.dataset_text_column
        texts = spacy_pipeline(
            texts=dataset[text_column], language=self.input_config.language
        )
        boundaries = []
        n_generated_sentences = []
        for text in texts:
            sentences = list([sent.text for sent in text.sents])
            self.workspace["human_spans"].append(sentences)
            self.workspace["positions"].append([])
            # If the text has more than one sentence, it can be used
            # to interleave generations w/ human sentences.
            max_boundaries_to_select = ceil(
                (len(sentences) - 1) * self.args["max_percentage_boundaries"]
            )
            accum = 0
            if len(sentences) > 1:
                for idx in range(len(sentences) - 1):
                    is_boundary = randint(0, 1)
                    if is_boundary and accum < max_boundaries_to_select:
                        gen_sents = randint(1, self.args["max_sentence_span"])
                        boundary = (sentences[idx], sentences[idx + 1])
                        boundaries.append(self._format_boundary(boundary))
                        self.workspace["positions"][-1].append(idx)
                        n_generated_sentences.append(gen_sents)
                        accum += 1
            # At this point:
            # (1) No boundaries have been selected:
            #     -> consider all the text as human
            # (2) >=1 boundary have been selected:
            #     -> the text could be interleaved (gen, human)
            # (3) The text has only one sentence:
            #     -> consider all the text as human
            self.workspace["num_boundaries"].append(accum)

        return {
            "n": list(map(str, n_generated_sentences)),
            "boundaries": boundaries,
        }
