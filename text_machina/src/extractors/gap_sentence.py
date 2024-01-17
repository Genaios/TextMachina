from math import ceil
from random import randint
from typing import Dict, List, Tuple

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class GapSentence(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("gap_sentence", {})
        self.positions = []
        self.human_spans = []
        self.num_boundaries = []

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return [" ".join(doc_sentences) for doc_sentences in self.human_spans]

    def _format_boundary(self, pair: Tuple[str, str]) -> str:
        return f"{pair[0]}\n{self.args['gap_token']}\n{pair[1]}"

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        dataset_column = self.args["dataset_column"]
        texts = spacy_pipeline(
            texts=dataset[dataset_column], language=self.input_config.language
        )
        boundaries = []
        n_generated_sentences = []
        for text in texts:
            sentences = list([sent.text for sent in text.sents])
            self.human_spans.append(sentences)
            self.positions.append([])
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
                        self.positions[-1].append(idx)
                        n_generated_sentences.append(gen_sents)
                        accum += 1
            # At this point:
            # (1) No boundaries have been selected:
            #     -> consider all the text as human
            # (2) >=1 boundary have been selected:
            #     -> the text could be interleaved (gen, human)
            # (3) The text has only one sentence:
            #     -> consider all the text as human
            self.num_boundaries.append(accum)

        return {
            "n": list(map(str, n_generated_sentences)),
            "boundaries": boundaries,
        }


if __name__ == "__main__":
    from datasets import Dataset

    from ..config import InputConfig

    input_config = InputConfig(
        quantity=1,
        domain="bla",
        dataset="bla",
        dataset_text_column="text",
        dataset_params={},
        template="Write {n} sentences to fill the gap marked as '____' between the following 2 sentences.Refrain to copy the sentences provided:\n{boundaries}\n\nSentences:",
        extractor="gap_sentence",
        language="en",
        extractor_args={
            "gap_sentence": {
                "dataset_column": "text",
                "max_sentence_span": 2,
                "max_percentage_boundaries": 0.2,
                "gap_token": "____",
            }
        },
    )
    dataset = Dataset.from_dict(
        {
            "text": [
                "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
                "Sentence 1.",
            ]
        }
    )
    ex = GapSentence(input_config, TaskType.DETECTION)
    res = ex.extract(dataset)
    print(res)
    print(ex.human_sentences)
    print(ex.positions)
    print(ex.num_boundaries)
