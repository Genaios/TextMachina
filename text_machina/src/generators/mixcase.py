from typing import Dict, List, Tuple

from datasets import Dataset, concatenate_datasets

from ..common.exceptions import DatasetGenerationError
from ..config import Config
from ..types import DetectionLabels, LabeledSpan, Placeholders
from .base import DatasetGenerator


class MixCaseDatasetGenerator(DatasetGenerator):
    """
    Dataset generator for the mixcase task type.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _interleave(
        self, generations: List[str]
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
        texts = []
        labels = []
        prev_sample = 0
        for idx, sample_boundaries in enumerate(
            self.prompter.extractor.workspace["num_boundaries"]
        ):
            # Text w/o sampled boundaries
            if sample_boundaries == 0:
                text = "".join(
                    self.prompter.extractor.workspace["human_spans"][idx]
                )
                texts.append(text)
                sample_labels = [
                    LabeledSpan(
                        start=0,
                        end=len(text),
                        label=DetectionLabels.HUMAN.value,
                    ).dict()
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

                sample_spans = self.prompter.extractor.workspace["human_spans"][
                    idx
                ]
                sample_positions = self.prompter.extractor.workspace[
                    "positions"
                ][idx]
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
                    ).dict()
                    gen_span = LabeledSpan(
                        start=gen_start,
                        end=gen_end,
                        label=DetectionLabels.GENERATED.value,
                    ).dict()
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
                        ).dict()
                    )

                texts.append(text)
            prev_sample += sample_boundaries
            labels.append(sample_labels)
        return texts, labels

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts.

        Args:
            generations (List[str]): list of generated texts.
            prompted_dataset (PromptedDataset): dataset with prompts and human texts.
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
        extractor = self.prompter.extractor
        texts, labels = self._interleave(generations)

        prev_sample = 0
        mixed_samples = []
        for idx, (text, sample_labels) in enumerate(zip(texts, labels)):
            sample_boundaries = extractor.workspace["num_boundaries"][idx]
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
            prev_sample += extractor.workspace["num_boundaries"][idx]

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
                        ).dict()
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
