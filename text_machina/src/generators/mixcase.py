import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

from datasets import Dataset, concatenate_datasets

from ..common import color_log, get_logger
from ..common.exceptions import DatasetGenerationError
from ..config import Config
from ..extractors import Extractor, SentenceGap, WordGap
from ..models.types import GENERATION_ERROR
from ..types import DetectionLabels, LabeledSpan, Placeholders
from .base import DatasetGenerator

_logger = get_logger(__name__)


class MixCaseDatasetGenerator(DatasetGenerator):
    """
    Dataset generator for the mixcase task type.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        if isinstance(self.prompter.extractor, (SentenceGap, WordGap)):
            packer: Type[MixCasePacker] = MixCaseGapPacker
        else:
            packer = MixCaseMaskPacker
        return packer(self.config, self.prompter.extractor)._pack(
            generations, **kwargs
        )


class MixCasePacker(ABC):
    """
    Base class for mixcase packers.
    """

    def __init__(self, config: Config, extractor: Extractor) -> None:
        self.config = config
        self.extractor = extractor

    @abstractmethod
    def _build_samples(
        self,
        generations: List[str],
    ) -> Tuple[List[str], List[List[Dict]]]:
        """
        Builds the samples to be packed in a dataset by preparing the texts
        and the labels according to the extractor (gap-based or mask-based).

        Args:
            generations (List[str]): list of generations.
            kwargs (Dict): additional keyword arguments.

        Returns:
            Tuple[List[str], List[List[Dict]]]: texts and labels to be added
                to the dataset.
        """
        ...

    @abstractmethod
    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts
        according to the extractor.

        Args:
            generations (List[str]): list of generated texts.
            kwargs: additional keyword arguments.

        Returns:
            Dataset: a dataset including all the texts.
        """
        ...


class MixCaseMaskPacker(MixCasePacker):
    """
    Packer for mixcase task type when using mask-based extractors.
    """

    def __init__(self, config: Config, extractor: Extractor) -> None:
        super().__init__(config=config, extractor=extractor)
        self.mask_regex = re.compile(
            rf"({self.extractor.args['mask_token']}-\d+)"
        )

    def _build_error_sample(self) -> Tuple[str, List[Dict]]:
        """
        Helper to build an error sample when required, e.g.,
        invalid JSON, missing masks, etc; that logs the cause of
        the resulting error (LLM uncapable always).

        Returns:
            Tuple[str, List[Dict]]: the text and labels of an error sample.
        """
        _logger.info(
            color_log(
                "The completion was not a valid JSON."
                " This error is related to the LLM capabilities,"
                " please, use another LLM if there are many"
                " errors of this type."
                " The text will be considered a generation error"
                f"`{GENERATION_ERROR}`",
                "bold_yellow",
            )
        )
        return GENERATION_ERROR, [
            LabeledSpan(
                start=0,
                end=len(GENERATION_ERROR),
                label=DetectionLabels.GENERATED.value,
            ).model_dump()
        ]

    def _build_sample(
        self, masked_text: str, parsed_generation: Dict[str, str]
    ) -> Tuple[str, List[Dict]]:
        """
        Builds a sample by reconstructing the masks in a text.

        Args:
            masked_text (str): a text with masks to be replaced.
            parsed_generation: (Dict[str, str]): dictionary mapping masks
                to texts, e.g. {"MASK-0": <text>, ...}

        Returns:
            Tuple[str, List[Dict]]: the text and labels of the sample.
        """
        # Replace masks in text and compute the generated labels
        # The following algorithm relies on masks sorted by their index.
        mask_completions = sorted(
            parsed_generation.items(), key=lambda x: int(x[0].split("-")[1])
        )
        sample_text = masked_text
        generated_labels = []
        for mask_token, completion in mask_completions:
            mask_match = re.search(rf"{mask_token}\b", sample_text)
            if mask_match is None:
                return self._build_error_sample()
            mask_position = mask_match.span()[0]
            # Ensure the completion does not append another mask token
            completion = self.mask_regex.sub("", completion)
            sample_text = re.sub(rf"({mask_token})\b", completion, sample_text)
            gen_end = mask_position + len(completion)
            generated_labels.append(
                LabeledSpan(
                    start=mask_position,
                    end=gen_end + 1,
                    label=DetectionLabels.GENERATED.value,
                ).model_dump()
            )

        # Compute human labels
        human_labels = []
        prev_generated_position = 0
        for idx in range(len(generated_labels)):
            generated_label = generated_labels[idx]
            if prev_generated_position < generated_label["start"]:
                human_labels.append(
                    LabeledSpan(
                        start=prev_generated_position,
                        end=generated_label["start"],
                        label=DetectionLabels.HUMAN.value,
                    ).model_dump()
                )
            prev_generated_position = generated_label["end"]

        # Add human label if the end of the text has not been
        # reached by the last generated span
        last_generated_position = generated_labels[-1]["end"]
        if last_generated_position < len(sample_text):
            human_labels.append(
                LabeledSpan(
                    start=last_generated_position,
                    end=len(sample_text),
                    label=DetectionLabels.HUMAN.value,
                ).model_dump()
            )

        # Join generated labels with human labels and merge overlappings.
        merged_labels = sorted(
            human_labels + generated_labels, key=lambda span: span["start"]
        )
        sample_labels = []
        while len(merged_labels):
            current_span = merged_labels.pop(0)
            current_label = current_span["label"]
            spans_with_same_label = []
            while (
                len(merged_labels)
                and merged_labels[0]["label"] == current_label
            ):
                spans_with_same_label.append(merged_labels.pop(0))

            if not spans_with_same_label:
                sample_labels.append(current_span)
            else:
                sample_labels.append(
                    LabeledSpan(
                        start=current_span["start"],
                        end=spans_with_same_label[-1]["end"],
                        label=current_label,
                    ).model_dump()
                )
        return sample_text, sample_labels

    def _build_samples(
        self,
        generations: List[str],
    ) -> Tuple[List[str], List[List[Dict]]]:
        """
        Reconstructs masked texts using completions to build mixcase samples.
        The `start` and `end` of the labels follow the Python [`start`, `end`)
        convention, i.e., including the `start` element and excluding `end`.
        For instance, given the text: "I like Apolo. I don't like Athenea",
        being the first sentence human-written and the second one generated,
        the labels will be:
        [
            {"start": 0, "end": 14, "label": "human"},
            {"start": 14, "end": 34, "label": "generated"},
        ]

        Args:
            generations (List[str]): list of generations

        Returns:
            Tuple[List[str], List[List[Dict]]]: the interleaved texts and
            the list of labels of each text.
        """
        texts, labels = [], []
        for generation, masked_text in zip(
            generations, self.extractor.workspace["masked_texts"]
        ):
            try:
                parsed_generation = json.loads(
                    generation[
                        generation.find("{") : generation.rfind("}") + 1
                    ].strip()
                )
            except json.decoder.JSONDecodeError:
                # If the LLM didn't generate a valid JSON output,
                # the sample will be considered as an error.
                sample_text, sample_labels = self._build_error_sample()
            else:
                # If the number of masks in the json do not match with
                # the number of masks in the text, the sample will be
                # considered as an error.
                masks_in_text = set(self.mask_regex.findall(masked_text))
                masks_in_completion = set(parsed_generation.keys())

                if len(masks_in_text.intersection(masks_in_completion)) != len(
                    masks_in_text
                ):
                    sample_text, sample_labels = self._build_error_sample()
                # Otherwise, the sample can be built.
                else:
                    sample_text, sample_labels = self._build_sample(
                        masked_text, parsed_generation
                    )
            finally:
                texts.append(sample_text)
                labels.append(sample_labels)
        return texts, labels

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts
        when using mask-based extractors (`sentence_masking`, `word_masking`, etc.)

        Args:
            generations (List[str]): list of generated texts.
            kwargs: additional keyword arguments.

        Returns:
            Dataset: a dataset including all the texts.
        """
        prompted_dataset = kwargs.get("prompted_dataset", None)
        if prompted_dataset is None:
            raise DatasetGenerationError(f"prompted_dataset not found: {self}")

        model_name = self.config.model.model_name
        domain = self.config.input.domain
        extractor_name = self.config.input.extractor
        texts, labels = self._build_samples(generations)

        generated_dataset = Dataset.from_list(
            [
                {
                    "prompt": prompt,
                    "text": text,
                    "label": label,
                    "model": model_name,
                    "domain": domain,
                    "extractor": extractor_name,
                }
                for prompt, text, label in zip(
                    prompted_dataset.prompted_texts, texts, labels
                )
            ]
        )

        human_dataset = Dataset.from_list(
            [
                {
                    "prompt": Placeholders.NO_PROMPT.value,
                    "text": text,
                    "label": [
                        LabeledSpan(
                            start=0,
                            end=len(text),
                            label=DetectionLabels.HUMAN.value,
                        ).model_dump()
                    ],
                    "model": DetectionLabels.HUMAN.value,
                    "domain": domain,
                    "extractor": Placeholders.NO_EXTRACTOR.value,
                }
                for text in prompted_dataset.human_texts
            ]
        )

        dataset = concatenate_datasets([human_dataset, generated_dataset])
        dataset = dataset.shuffle()

        return dataset


class MixCaseGapPacker(MixCasePacker):
    """
    Packer for mixcase task type when using gap-based extractors.
    """

    def __init__(self, config: Config, extractor: Extractor) -> None:
        super().__init__(config=config, extractor=extractor)

    def _build_samples(
        self,
        generations: List[str],
    ) -> Tuple[List[str], List[List[Dict]]]:
        """
        Interleaves generated and human spans to build mixcase samples.
        The `start` and `end` of the labels follow the Python [`start`, `end`)
        convention, i.e., including the `start` element and excluding `end`.
        For instance, given the text: "I like Apolo. I don't like Athenea",
        being the first sentence human-written and the second one generated,
        the labels will be:
        [
            {"start": 0, "end": 14, "label": "human"},
            {"start": 14, "end": 34, "label": "generated"},
        ]

        Args:
            generations (List[str]): list of generations

        Returns:
            Tuple[List[str], List[List[Dict]]]: the interleaved texts and
            the list of labels of each text.
        """
        texts, labels = [], []
        prev_sample = 0
        for idx, sample_boundaries in enumerate(
            self.extractor.workspace["num_boundaries"]
        ):
            # Text w/o sampled boundaries
            if sample_boundaries == 0:
                text = "".join(self.extractor.workspace["human_spans"][idx])
                texts.append(text)
                sample_labels = [
                    LabeledSpan(
                        start=0,
                        end=len(text),
                        label=DetectionLabels.HUMAN.value,
                    ).model_dump()
                ]
            # Text w/ sampled boundaries
            else:
                sample_generations = generations[
                    prev_sample : prev_sample + sample_boundaries
                ]
                # Add a whitespace after generations
                # to be concatenated with the suffix
                sample_generations = [
                    f"{generation} " for generation in sample_generations
                ]

                sample_spans = self.extractor.workspace["human_spans"][idx]
                sample_positions = self.extractor.workspace["positions"][idx]
                sample_labels = []
                added = 0
                prev_label_pos = 0
                # Interleave the generations in the positions determined
                # by the extractor, and computes the labeled spans.
                for i, position in enumerate(sample_positions):
                    # Interleave generation.
                    sample_spans.insert(
                        position + 1 + added, sample_generations[i]
                    )
                    # The generated span starts just after
                    # the prefix until the position `position`.
                    gen_start = len(
                        "".join(sample_spans[: position + 1 + added])
                    )
                    gen_end = gen_start + len(sample_generations[i])
                    # The human span starts from the previous generated span
                    # if exists (prev_label_pos != -1).
                    human_span = LabeledSpan(
                        start=prev_label_pos,
                        end=gen_start,
                        label=DetectionLabels.HUMAN.value,
                    ).model_dump()
                    gen_span = LabeledSpan(
                        start=gen_start,
                        end=gen_end,
                        label=DetectionLabels.GENERATED.value,
                    ).model_dump()
                    sample_labels.append(human_span)
                    sample_labels.append(gen_span)
                    added += 1
                    prev_label_pos = gen_end
                text = "".join(sample_spans)
                # Fill the labels if them do not cover all the text yet.
                # The last span is always human by construction.
                if int(sample_labels[-1]["end"]) < len(text):
                    sample_labels.append(
                        LabeledSpan(
                            start=sample_labels[-1]["end"],
                            end=len(text),
                            label=DetectionLabels.HUMAN.value,
                        ).model_dump()
                    )

                texts.append(text)
            prev_sample += sample_boundaries
            labels.append(sample_labels)
        return texts, labels

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts
        when using gap-based extractors (`sentence_gap`, `word_gap`, etc.)

        Args:
            generations (List[str]): list of generated texts.
            kwargs: additional keyword arguments.

        Returns:
            Dataset: a dataset including all the texts.
        """
        prompted_dataset = kwargs.get("prompted_dataset", None)
        if prompted_dataset is None:
            raise DatasetGenerationError(f"prompted_dataset not found: {self}")

        model_name = self.config.model.model_name
        domain = self.config.input.domain
        extractor_name = self.config.input.extractor
        texts, labels = self._build_samples(generations)

        prev_sample = 0
        mixed_samples = []
        for idx, (text, sample_labels) in enumerate(zip(texts, labels)):
            sample_boundaries = self.extractor.workspace["num_boundaries"][idx]
            prompt = prompted_dataset.prompted_texts[
                prev_sample : prev_sample + sample_boundaries
            ] or [Placeholders.NO_PROMPT.value]

            mixed_samples.append(
                {
                    "prompt": prompt,
                    "text": text,
                    "label": sample_labels,
                    "model": model_name,
                    "domain": domain,
                    "extractor": extractor_name,
                }
            )
            prev_sample += self.extractor.workspace["num_boundaries"][idx]

        mixed_dataset = Dataset.from_list(mixed_samples)

        human_dataset = Dataset.from_list(
            [
                {
                    "prompt": [Placeholders.NO_PROMPT.value],
                    "text": text,
                    "label": [
                        LabeledSpan(
                            start=0,
                            end=len(text),
                            label=DetectionLabels.HUMAN.value,
                        ).model_dump()
                    ],
                    "model": DetectionLabels.HUMAN.value,
                    "domain": domain,
                    "extractor": Placeholders.NO_EXTRACTOR.value,
                }
                for text in prompted_dataset.human_texts
            ]
        )

        dataset = concatenate_datasets([human_dataset, mixed_dataset])
        dataset = dataset.shuffle()

        return dataset
