from enum import Enum
from typing import List

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectionLabels(Enum):
    """
    Labels for detection tasks.
    """

    GENERATED: str = "generated"
    HUMAN: str = "human"


class Placeholders(Enum):
    """
    Placeholders to be used across TextMachina.
    """

    NO_PROMPT: str = "NO-PROMPT"
    NO_EXTRACTOR: str = "NO-EXTRACTOR"


class Prompt(BaseModel):
    """
    Wrapper for a prompt.
    """

    template: str
    extractor: str


class PromptedDataset(BaseModel):
    """
    Wrapper for a prompted dataset used to generate MGT texts.
    """

    prompted_texts: List[str]
    human_texts: List[str]
    human_images: List[Image] = []

    # TODO: Change this, e.g., instead of using Image type, storing bytes
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskType(str, Enum):
    DETECTION: str = "detection"
    ATTRIBUTION: str = "attribution"
    BOUNDARY: str = "boundary"
    MIXCASE: str = "mixcase"


class Modality(str, Enum):
    IMAGE: str = "image"
    TEXT: str = "text"
    AUDIO: str = "audio"
    VIDEO: str = "video"


class LabeledSpan(BaseModel):
    start: int = Field(ge=0)
    end: int
    label: str

    @model_validator(mode="after")
    def check_valid_positions(self) -> "LabeledSpan":
        if not (self.start < self.end):
            raise ValueError(
                "`start` must be lower than `end` in a LabeledSpan"
                f" (`start`={self.start}, `end`={self.end})"
            )
        return self
