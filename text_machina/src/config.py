from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from yaml import full_load, safe_load

from .types import TaskType


class InputConfig(BaseModel):
    """
    Wrapper for the input_config field.
    """

    quantity: int = Field(gt=0, description="Number of samples to generate.")
    domain: str = Field(description="Domain of a dataset.")
    dataset: str = Field(description="Name (HF Hub) or path to the dataset.")
    dataset_text_column: str = Field(
        description="Name of column in the dataset containing the text."
    )
    dataset_params: Dict[str, Any] = Field(description="Arguments to load the dataset.")
    template: str = Field(description="Template for the generations.")
    extractor: str = Field(description="Extractor name.")
    extractors_list: List[str] = Field(
        default=[],
        description=("List of extractors to be used" " with the `combined` extractor."),
        validate_default=True,
    )
    random_sample_human: bool = Field(
        default=False,
        desc=(
            "Whether to randomly sample human texts or use"
            " the same ones used to generate MGT"
        ),
    )
    max_input_tokens: int = Field(
        default=256,
        gt=0,
        desc=(
            "Maximum token length to be distributed across the"
            " prompt inputs extracted with the extractors."
        ),
    )
    extractor_args: Dict[str, Dict[str, Any]] = Field(
        default={}, desc="Extractors-specific arguments."
    )
    language: str = Field(
        default="en",
        desc="Language of the dataset used.",
        validate_default=True,
    )

    @field_validator("language")
    @classmethod
    def language_must_be_iso639(cls, language: str) -> str:
        import pycountry

        allowed_languages = [
            lang.alpha_2 for lang in pycountry.languages if hasattr(lang, "alpha_2")
        ] + ["multilingual"]

        if language not in allowed_languages:
            from .common import InvalidLanguage

            raise InvalidLanguage()

        return language

    @field_validator("extractor")
    @classmethod
    def extractor_must_exist(cls, extractor: str) -> str:
        from .extractors import EXTRACTORS

        if extractor not in EXTRACTORS.keys():
            from .common import InvalidExtractor

            raise InvalidExtractor(extractor)

        return extractor

    @field_validator("extractors_list")
    @classmethod
    def not_empty_list_in_combined(
        cls, extractors_list: List[str], info: ValidationInfo
    ) -> List[str]:
        if info.data["extractor"] == "combined" and not extractors_list:
            from .common import CombinedEmptyExtractors

            raise CombinedEmptyExtractors()

        return extractors_list


class ModelConfig(BaseModel):
    """
    Wrapper for the input_config field.
    """

    provider: str = Field(description="Provider of text generation models.")
    model_name: str = Field(description="Name of a text generation model.")
    threads: int = Field(
        default=8,
        gt=0,
        description="Number of threads to use in `generate_completions`",
    )
    api_type: Literal["CHAT", "COMPLETION"] = Field(
        default="COMPLETION",
        description=(
            "API type for providers that allows chat and completion endpoints."
            "This arg must be `CHAT` or `COMPLETION` and must be according to"
            "the model used:\n"
            "- `CHAT`: for chat completion endpoints.\n"
            "- `COMPLETION` for traditional completion endpoints.\n"
            "For instance, GPT-4 in OpenAI can only be used with `CHAT`."
        ),
    )
    # Allow extra args and avoid protected naming conflicts
    model_config = ConfigDict(extra="allow", protected_namespaces=(""))

    @field_validator("provider")
    @classmethod
    def provider_must_exist(cls, provider: str) -> str:
        from .common import InvalidProvider
        from .models import MODELS

        if provider not in MODELS.keys():
            raise InvalidProvider(provider)
        return provider


class Config(BaseModel):
    """
    Wrapper for the config.
    """

    path: Optional[Path] = None
    task_type: TaskType
    input: InputConfig
    model: ModelConfig
    generation: Dict[str, Any]

    # Avoid protected naming conflicts
    model_config = ConfigDict(protected_namespaces=(""))

    @classmethod
    def load_config(
        cls: Type["Config"],
        path: Path,
        task_type: TaskType,
        max_generations: Optional[int] = None,
    ) -> "Config":
        config = safe_load(path.open("r"))

        if max_generations:
            config["input_config"]["quantity"] = min(
                max_generations, config["input_config"]["quantity"]
            )

        input = InputConfig(**config["input_config"])
        model = ModelConfig(**config["model_config"])
        generation = config["generation_config"]

        return cls(
            path=path,
            task_type=task_type,
            input=input,
            model=model,
            generation=generation,
        )

    @classmethod
    def load_configs(
        cls: Type["Config"],
        path: Path,
        task_type: TaskType,
        max_generations: Optional[int] = None,
    ) -> List["Config"]:
        path_iterator = (
            [path]
            if path.suffix in {".yml", ".yaml"}
            else chain(path.rglob("*.yml"), path.rglob("*.yaml"))
        )

        if not path_iterator:
            raise ValueError("The provided path does not contain any YML files", path)

        return [
            cls.load_config(
                path=p, task_type=task_type, max_generations=max_generations
            )
            for p in path_iterator
        ]

    def safe_model_name(self) -> str:
        return self.model.model_name.split("/")[-1].replace("_", "-")

    def safe_dataset_name(self) -> str:
        return self.input.dataset.split("/")[-1].replace("_", "-")

    def safe_domain_name(self) -> str:
        return self.input.domain.replace("/", "-").replace("_", "-")


def parse_metrics_config(path: Path) -> Tuple[List[str], Dict]:
    """
    Parses a metrics config.

    Args:
        path (Path): the metric config path to parse.

    Returns:
        Tuple[List[str], Dict]: a tuple of structure (list of metrci names, args).
    """
    config = full_load(path.open("r"))
    metrics = config["metrics_to_run"]
    del config["metrics_to_run"]

    return metrics, config
