from math import ceil
from random import choice, sample, uniform
from typing import Any, Dict, List

from datasets import Dataset

from ..common import color_log, get_logger
from ..common.exceptions import ExtractorInvalidArgs
from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline

_logger = get_logger(__name__)


class SentenceMasking(Extractor):
    """
    Extractor that fills the prompt template with a text with
    masked sentences and the LLM has to generate all the masked
    sentences.

    This extractor needs two template placeholders:
        - {masked_text}: will be filled with a text with masked sentences.

    This extractor allows to pass the following arguments in the
    `extractor_args` field from the config:
        - mask_token (str): mask token, e.g., "MASK". Several masks in a text
            will be appended with the index, e.g. "MASK-0"
        - percentage_range (List[float]): range delimiting the percentage
            of sentences to be masked. At least one sentence will be
            always masked.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        args: Dict[str, Any] = input_config.extractor_args.get(
            "sentence_masking", {}
        )
        workspace: Dict[str, Any] = {"masked_texts": []}
        super().__init__(input_config, task_type, workspace, args)

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

    def check_valid_args(self):
        mandatory_args = ["mask_token", "percentage_range"]
        for mandatory_arg in mandatory_args:
            if mandatory_arg not in self.args:
                raise ExtractorInvalidArgs(
                    self.__class__.__name__, mandatory_args
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
            sentences = [sent.text_with_ws for sent in text.sents]
            # In case of 1 sentences, avoid masking it,
            # just add a mask sentence to right or left randomly.
            if len(sentences) == 1:
                mask_position = choice([0, 1])
                fmt_masked_token = self._format_mask_token(0)
                sentences.insert(mask_position, fmt_masked_token)
                masked_texts.append("".join(sentences).strip())
            else:
                percentage_masks = uniform(*self.args["percentage_range"])
                # Min to avoid masking all the sentences.
                sents_to_mask = min(
                    len(sentences) - 1, ceil(percentage_masks * len(sentences))
                )
                positions = sorted(sample(range(len(sentences)), sents_to_mask))
                for mask_idx, sent_idx in enumerate(positions):
                    sentences[sent_idx] = self._format_mask_token(mask_idx)
                masked_texts.append("".join(sentences).strip())
        self.workspace["masked_texts"] = masked_texts
        return {"masked_text": masked_texts}
