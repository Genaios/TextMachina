from math import ceil
from random import choice, randint, uniform
from typing import Dict, List

from datasets import Dataset

from ..common import color_log, get_logger
from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline

_logger = get_logger(__name__)


class WordMasking(Extractor):
    """
    Extractor that fills the prompt template with a text with
    masked word spans and the LLM has to generate all the masked
    word spans.

    This extractor needs two template placeholders:
        - {masked_text}: will be filled with a text with masked word spans.

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - mask_token (str): mask token, e.g., "MASK". Several masks in a text
            will be appended with the index, e.g. "MASK-0"
        - percentage_range (List[float]): range delimiting the percentage
            of word spans to be masked. At least one word span will be
            always masked.
        - span_length_range (List[int]): range where to sample the length
            of each masked span.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("word_masking", {})
        self.workspace = {"masked_texts": []}

        _logger.warn(
            color_log(
                f"You are using the `{self.__class__.__name__}` extractor."
                " Consider that few models like GPT-4 can work properly with this"
                " type of generation. Models must be:\n"
                "1) Capable of generating proper JSON.\n"
                "2) Capable enough to generate all the masks.",
                "bold_yellow",
            )
        )

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return human_texts

    def _format_mask_token(
        self, idx: int, add_period: bool = False, add_whitespace: bool = True
    ) -> str:
        period = "." if add_period else ""
        whitespace = " " if add_whitespace else ""
        return f"{self.args['mask_token']}-{idx}{period}{whitespace}"

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        text_column = self.input_config.dataset_text_column
        texts = spacy_pipeline(
            texts=dataset[text_column], language=self.input_config.language
        )
        masked_texts = []
        for text in texts:
            words = [word.text_with_ws for word in text]
            percentage_masks = uniform(*self.args["percentage_range"])
            spans_to_mask = min(
                ceil((len(words) - 1) / max(self.args["span_length_range"])),
                ceil(
                    percentage_masks
                    * (len(words) / max(self.args["span_length_range"]))
                ),
            )

            sampled_positions: List[int] = []
            positions = list(range(len(words)))
            # Sample N words with at least max(args["span_length_range"])
            # positions of separation
            while (
                len(sampled_positions) < spans_to_mask
                and len(positions) >= spans_to_mask
            ):
                start_position = choice(positions)
                valid_positions = [
                    position
                    for position in positions
                    if abs(position - start_position)
                    >= max(self.args["span_length_range"])
                ]
                if valid_positions:
                    chosen_position = choice(valid_positions)
                    sampled_positions.append(chosen_position)
                    positions.remove(chosen_position)

            # Sample lengths and mask spans of the text
            sampled_positions = sorted(sampled_positions)
            span_lengths = [
                randint(*self.args["span_length_range"])
                for _ in range(len(sampled_positions))
            ]
            masked_words = []
            prev_position = 0
            mask_idxs = 0
            for position, length in zip(sampled_positions, span_lengths):
                masked_words.extend(words[prev_position:position])
                masked_words.append(self._format_mask_token(mask_idxs))
                prev_position = position + length
                mask_idxs += 1
            masked_texts.append("".join(masked_words))
        self.workspace["masked_texts"] = masked_texts
        return {"masked_text": masked_texts}
