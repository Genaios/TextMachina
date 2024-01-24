from .hf_local import HuggingFaceLocalTokenizer


class HuggingFaceRemoteTokenizer(HuggingFaceLocalTokenizer):
    """
    Tokenizer for HuggingFace models served in Inference API or Endpoints.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
