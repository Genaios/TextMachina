from typing import List


class TextMachinaError(Exception):
    """
    Base class for TextMachina exceptions.
    """

    ...


class MissingIntegrationError(TextMachinaError):
    """
    Raised when an integration can't be used due to missing dependencies.
    """

    def __init__(self, integration: str):
        self.integration = integration
        msg = (
            f"'{self.integration}' integration can't be used. Please, install"
            f" the corresponding integration:"
            f" `pip install text-machina[{integration}]`."
        )
        super().__init__(msg)


class MissingMetricError(TextMachinaError):
    """
    Raised when a metric can't be used due to missing dependencies.
    """

    def __init__(self, metric: str):
        self.metric = metric
        msg = (
            f"'{self.metric}' metric can't be used. Please, install"
            f" the extra exploration packages"
            f" `pip install text-machina[explore]`."
        )
        super().__init__(msg)


class InvalidInferenceServer(TextMachinaError):
    """
    Raised when an inference server type is not valid
    for the InferenceServerModel.
    """

    def __init__(self, inference_server: str, allowed: List[str]):
        self.inference_server = inference_server
        self.allowed = allowed
        msg = (
            f"Inference server `{self.inference_server}` is not allowed."
            f" Please use one from {', '.join(allowed)}"
        )
        super().__init__(msg)


class InvalidTaskTypeForMetric(TextMachinaError):
    """
    Raised when a metric can't be used due to inadequate task type.
    """

    def __init__(self, metric: str, task_type: str):
        self.metric = metric
        self.task_type = task_type
        msg = f"'{self.metric}' metric cannot be used for task type {self.task_type},"
        super().__init__(msg)


class InvalidProvider(TextMachinaError):
    """
    Raised when a provider is not allowed.
    """

    def __init__(self, provider: str):
        self.provider = provider
        from ..models import MODELS

        msg = (
            f"`provider={provider}` is not allowed. Use one of:"
            f" {', '.join(list(MODELS.keys()))} and install the"
            " corresponding integration."
        )
        super().__init__(msg)


class InvalidMetric(TextMachinaError):
    """
    Raised when a metric is not allowed.
    """

    def __init__(self, metric: str):
        self.metric = metric
        from ..metrics import METRICS

        msg = (
            f"`metric={metric}` is not allowed. Use one of:"
            f" {', '.join(list(METRICS.keys()))} and install"
            " exploration packages."
        )
        super().__init__(msg)


class InvalidExtractor(TextMachinaError):
    """
    Raised when an extractor is not allowed.
    """

    def __init__(self, extractor: str):
        self.extractor = extractor
        from ..extractors import EXTRACTORS

        msg = (
            f"`extractor={self.extractor}` is not allowed. Use one of:"
            f" {', '.join(list(EXTRACTORS.keys()))}."
        )
        super().__init__(msg)


class CombinedEmptyExtractors(TextMachinaError):
    """
    Raised when `combined` extractor is called without a list of extractors.
    """

    def __init__(self):
        msg = (
            "Specify a non-empty list of extractors in the `extractors_list`"
            " field of your config to use the `combined` extractor."
        )
        super().__init__(msg)


class InvalidModelParam(TextMachinaError):
    """
    Raised when a parameter for a text generation model is invalid.
    """

    def __init__(self, msg: str):
        super().__init__(msg)


class DatasetGenerationError(TextMachinaError):
    """
    Raised when an error ocurrs in dataset generators.
    """

    def __init__(self, msg: str):
        super().__init__(msg)


class UnsupportedMetricParam(TextMachinaError):
    """
    Raised when a metric param cannot be used.

    This is not for cases where an incorrect parameter name is provided,
    rather when a parameter is correct but cannot be used.

    See `..metrics.mauve.MAUVEMetric` for an example.
    """

    def __init__(self, param_name: str, metric_name: str):
        self.param_name = param_name
        self.metric_name = metric_name
        msg = f"Parameter {self.param_name} cannot be used for metric {metric_name}"
        super().__init__(msg)


class InvalidLanguage(TextMachinaError):
    """
    Raised when a language is not a valid ISO 639-1 code.
    """

    def __init__(self):
        msg = "`language` must be a valid ISO 639-1 code."
        super().__init__(msg)


class InvalidSpacyModel(TextMachinaError):
    """
    Raised when an Spacy model does not exist for a language.
    """

    def __init__(self, language: str):
        self.language = language
        msg = f"Spacy does not have a model for `language={self.language}`"
        super().__init__(msg)


class ExtractorEmptyColumns(TextMachinaError):
    """
    Raised when an extractor returns empty lists in any field.
    """

    def __init__(self, extractor: str, field: str):
        self.field = field
        msg = f"The extractor {extractor} returned empty '{field}'."
        super().__init__(msg)


class ExtractorInvalidArgs(TextMachinaError):
    """
    Raised when mandatory arguments for an
    extractor are invalid or not provided.
    """

    def __init__(self, extractor: str, extractor_args: List[str]):
        self.extractor = extractor
        self.extractor_args = extractor_args
        msg = (
            f"When using the extractor {extractor}, you must ensure that"
            f" the following arguments: {', '.join(self.extractor_args)}"
            " are valid."
        )
        super().__init__(msg)
