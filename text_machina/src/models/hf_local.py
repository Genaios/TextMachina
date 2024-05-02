from typing import Dict, List

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ModelConfig
from .base import TextGenerationModel
from .types import QUANTIZATION_CONFIGS, CompletionType


class HuggingFaceLocalModel(TextGenerationModel):
    """
    Generates completions using HuggingFace's models locally deployed.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.model_name = getattr(self.model_config, "model_name")
        self.quantization = getattr(self.model_config, "quantization", "none")
        self.batch_size = getattr(self.model_config, "batch_size", 8)
        self.device = getattr(self.model_config, "device", "cpu")

        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def generate_completion(self, prompt: str, generation_config: Dict) -> str:
        """
        Override `generate_completions` for completeness.
        This method is not used, since generations are done
        with batches using `generate_completions`.
        """

        if self.model_config.api_type == CompletionType.CHAT:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        tokenized = self.tokenizer(
            prompt, truncation=True, padding=True, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)
        completion = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_config,
        )[0]
        return self.tokenizer.decode(
            completion[len(input_ids[0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def generate_completions(
        self,
        prompts: List[str],
        generation_config: Dict,
    ) -> List[str]:
        """
        Overriden method to generate completions using
        HuggingFace's `generate` method with batches
        """
        if self.model_config.api_type == CompletionType.CHAT:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts
            ]

        tokenized_prompts = self.tokenizer(
            prompts, truncation=True, padding=True, return_tensors="pt"
        )
        completions = []
        for batch_idx in tqdm(
            range(0, len(prompts), self.batch_size),
            desc=f"Generating locally with {self.model_name}",
        ):
            input_ids = tokenized_prompts["input_ids"][
                batch_idx : batch_idx + self.batch_size
            ].to(self.device)
            attention_mask = tokenized_prompts["attention_mask"][
                batch_idx : batch_idx + self.batch_size
            ].to(self.device)
            batch_completions = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_config,
            )
            for idx, completion in enumerate(batch_completions):
                completions.append(
                    self.tokenizer.decode(
                        completion[len(input_ids[idx]) :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
        return completions

    def __load_model(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **QUANTIZATION_CONFIGS[self.quantization],
        )
        if self.quantization == "none":
            model.to(self.device)
        return model

    def __load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return tokenizer
