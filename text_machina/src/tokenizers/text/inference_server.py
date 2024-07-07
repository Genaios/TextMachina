from .hf_local import HuggingFaceLocalTokenizer


class InferenceServerTokenizer(HuggingFaceLocalTokenizer):
    """
    Tokenizer for models deployed on inference servers
    like VLLM or TRT. This tokenizer assumes the tokenizers
    of the deployed models are available on the HF hub.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
