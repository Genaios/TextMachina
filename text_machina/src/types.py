from enum import Enum
from typing import List

from pydantic import BaseModel


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


class TaskType(str, Enum):
    DETECTION: str = "detection"
    ATTRIBUTION: str = "attribution"
    BOUNDARY: str = "boundary"
    MIXCASE: str = "mixcase"


class LabeledSpan(BaseModel):
    start: int
    end: int
    label: str
