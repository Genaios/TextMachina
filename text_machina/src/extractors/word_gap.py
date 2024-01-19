from math import ceil
from random import randint
from typing import Dict, List, Tuple

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class WordGap(Extractor):
    """
    Extractor that fills the prompt template with a boundary of
    two word spans (left-side and right-side of a sampled word),
    and with the number of words the LLM has to generate in
    between the boundary word spans.

    This extractor needs two template placeholders:
        - {n}: will be filled with the number of words to generate
               between the boundary words.
        - {boundaries}: will be filled with the boundary words separated
                        by the gap token and newlines.
                        E.g., "words1 ____ words2"

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - gap_token (str): gap token, e.g., "____"
        - max_percentage_boundaries (float): max percentage of
          boundaries to sample from a text. In a text of N words,
          there will be N-1 possible boundaries of two word spans.
        - max_word_span (int): max number of words to be generated
          between the boundary words.
        - range_boundary_size (List[float, float]): range where
          to sample the length of the word spans in the boundaries.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("word_gap", {})
        self.workspace = {
            "positions": [],
            "human_spans": [],
            "num_boundaries": [],
        }

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return [
            "".join(doc_words) for doc_words in self.workspace["human_spans"]
        ]

    def _format_boundary(self, pair: Tuple[str, str]) -> str:
        return f"{pair[0]} {self.args['gap_token']} {pair[1]}"

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        text_column = self.input_config.dataset_text_column
        texts = spacy_pipeline(
            texts=dataset[text_column], language=self.input_config.language
        )
        boundaries = []
        n_generated_words = []
        for text in texts:
            words = [word.text_with_ws for word in text]
            self.workspace["human_spans"].append(words)
            self.workspace["positions"].append([])
            # If the text has more than one word, it can be used
            # to interleave generations w/ human words.
            max_boundaries_to_select = ceil(
                (len(words) - 1) * self.args["max_percentage_boundaries"]
            )
            accum = 0
            if len(words) > 1:
                idx = 0
                while idx < len(words) - 1:
                    is_boundary = randint(0, 1)
                    boundary_size = 0
                    if is_boundary and accum < max_boundaries_to_select:
                        gen_sents = randint(1, self.args["max_word_span"])
                        boundary_size = randint(
                            *self.args["range_boundary_size"]
                        )
                        boundary = (
                            "".join(
                                words[max(0, idx - boundary_size) : idx + 1]
                            ),
                            "".join(words[idx + 1 : idx + 1 + boundary_size]),
                        )
                        boundaries.append(self._format_boundary(boundary))
                        self.workspace["positions"][-1].append(idx)
                        n_generated_words.append(gen_sents)
                        accum += 1
                    # Move far away to avoid coherence conflicts
                    # between boundary generations
                    idx += 1 + boundary_size

            # At this point:
            # (1) No boundaries have been selected:
            #     -> consider all the text as human
            # (2) >=1 boundary have been selected:
            #     -> the text could be interleaved (gen, human)
            # (3) The text has only one word:
            #     -> consider all the text as human
            self.workspace["num_boundaries"].append(accum)

        return {
            "n": list(map(str, n_generated_words)),
            "boundaries": boundaries,
        }
