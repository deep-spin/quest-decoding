from typing import List
from quest.reward.base import Reward
import gc

from transformers import pipeline
from quest.utils.logger import fix_loggers
from quest.utils.math import clamp_logit
from quest.utils.data import get_loader
import numpy as np
from tqdm import tqdm

from quest.model.base import LanguageModel
from langchain.prompts import PromptTemplate
from trl import AutoModelForCausalLMWithValueHead
from transformers.utils import cached_file

# Ignore specific UserWarning from transformers library
# Configure the transformers logger

fix_loggers(name="transformers")

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from typing import Dict
from torch import nn


class RewardModel(Reward):

    # applies the model only on outputs r(y)
    """
    RewardModel class represents a reward model based on a pre-trained Hugging Face model.

    Args:
        model_path (str): The path to the pre-trained model.
        batch_size (int, optional): The batch size for inference. Defaults to 32.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.

    Attributes:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to use for inference.

    Methods:
        evaluate(candidates: List[str]) -> List[float]:
            Evaluates a list of candidate sequences and returns a list of reward values.

    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: int = 0,
        task: str = "text-classification",
        clamp: float = 40,
        dtype=torch.bfloat16,
        use_flash_attention: bool = True,
        device_count=1,
    ):

        super().__init__(f"rm:{model_path}")

        self.batch_size = batch_size
        self.device = device
        """self.model = pipeline(
            task,
            model=model_path,
            device=device,
        )"""
        self.kwargs = {"batch_size": self.batch_size}
        self.clamp = clamp

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )  # THIS IS ACTUALLY REALLY IMPORTANT :) THIS HIDDEN NIGHTMARE DONT USE EOS. - w/ AR models in batch we may have padding in the beginig
            self.tokenizer.pad_token = self.tokenizer.bos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=dtype,
            pad_token_id=self.tokenizer.pad_token_id,
            use_flash_attention_2=use_flash_attention,
            # device_map="auto",
        )

        if device_count > 1:

            self.model = nn.DataParallel(
                self.model, device_ids=(np.arange(device_count) + self.device).tolist()
            )

        self.device = torch.device(
            f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        candidates: List[str],
        use_tqdm=False,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int]): The list of indices of accepted candidates.
            batch_size (int, optional): The batch size for inference. Defaults to 32.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        loader = get_loader(
            candidates,
            self.tokenizer,
            use_tqdm=use_tqdm,
            batch_size=self.batch_size,
        )

        rewards = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = np.clip(
                    outputs.logits[:, 0].float().cpu().numpy(),
                    -self.clamp,
                    self.clamp,
                ).tolist()

                del outputs, input_ids, attention_mask
                gc.collect()
                torch.cuda.empty_cache()

                rewards.extend(logits)

        return rewards


class ContextualRewardModel(RewardModel):

    # applies the model only on outputs r(y)
    """
    RewardModel class represents a reward model based on a pre-trained Hugging Face model.

    Args:
        model_path (str): The path to the pre-trained model.
        batch_size (int, optional): The batch size for inference. Defaults to 32.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.

    Attributes:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to use for inference.

    Methods:
        evaluate(candidates: List[str]) -> List[float]:
            Evaluates a list of candidate sequences and returns a list of reward values.

    """

    def __init__(self, **rm_kwargs):
        super().__init__(**rm_kwargs)
        self.name = "c" + self.name
        self.prompt_txt = None

    def set_context(self, context: List[str]):
        self.prompt_txt = context

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int] = None,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        if self.prompt_txt is None:
            context_candidates = candidates
        else:

            context_candidates = [
                prompt + str2 for prompt, str2 in zip(self.prompt_txt, candidates)
            ]

        return super().evaluate(context_candidates, **kwargs)


class LLMReward(Reward):
    def __init__(self, llm: LanguageModel):
        super().__init__("llm:" + llm.model_path)
        self.llm = llm

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int] = None,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        _, scores = self.llm.evaluate_continuation(
            len(candidates) * [[]],
            candidates,
        )

        return list(map(np.mean, scores))


class TemplateLLMReward(LLMReward):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_context(self, context: List[PromptTemplate]):
        self.prompt_txt = context

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int] = None,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        if accepted_indices is not None:
            prompts = [self.prompt_txt[i] for i in accepted_indices]
        else:
            prompts = self.prompt_txt

        context_candidates = [
            template.format(text=str2) for template, str2 in zip(prompts, candidates)
        ]

        return super().evaluate(context_candidates, **kwargs)


class PromptedLLMReward(TemplateLLMReward):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prefix_prompt = PromptTemplate.from_template("{c}{text}")

    def set_context(self, context: List[str]):
        super().set_context([self.prefix_prompt.partial(c=c) for c in context])


class ValueHead(Reward):

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: int = 0,
        task: str = "text-classification",
        clamp: float = 40,
        dtype=torch.bfloat16,
        use_flash_attention: bool = True,
        device_count=1,
    ):

        super().__init__("vh:" + model_path)

        self.batch_size = batch_size
        self.device = device

        self.kwargs = {"batch_size": self.batch_size}
        self.clamp = clamp

        from transformers import (
            AutoTokenizer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )  # THIS IS ACTUALLY REALLY IMPORTANT :) THIS HIDDEN NIGHTMARE DONT USE EOS. - w/ AR models in batch we may have padding in the beginig
            self.tokenizer.pad_token = self.tokenizer.bos_token

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # pad_token_id=self.tokenizer.pad_token_id,
            use_flash_attention_2=use_flash_attention,
            # device_map="auto",
        ).to(dtype)

        vhead_params = torch.load(
            model_path + "/value_head.bin", map_location="cpu", weights_only=True
        )

        self.model.v_head.summary.weight = torch.nn.Parameter(
            vhead_params["v_head.summary.weight"]
        )
        self.model.v_head.summary.bias = torch.nn.Parameter(
            vhead_params["v_head.summary.bias"]
        )

        if device_count > 1:

            self.model = nn.DataParallel(
                self.model, device_ids=(np.arange(device_count) + self.device).tolist()
            )

        self.device = torch.device(
            f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        self.prompt_txt = None

    def _evaluate(
        self,
        candidates: List[str],
        use_tqdm=False,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int]): The list of indices of accepted candidates.
            batch_size (int, optional): The batch size for inference. Defaults to 32.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """
        loader = get_loader(
            candidates,
            self.tokenizer,
            use_tqdm=use_tqdm,
            batch_size=self.batch_size,
            max_length=1024,
        )

        rewards = []

        with torch.no_grad():
            for batch in loader:
                # sj_text = self.tokenizer.decode(batch["input_ids"][0])
                # si_text = self.tokenizer.decode(si)
                # print(sj_text)
                # print("-" * 100)
                # print(candidates[0])

                # (attention_mask[0] == 0).sum()
                # self.tokenizer.decode(batch["input_ids"][0][875:])
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits, _, values = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # return_dict=True,
                )

                # outputs

                values = values[:, -1]

                logits = np.clip(
                    values.flatten().float().cpu().numpy(),
                    -self.clamp,
                    self.clamp,
                ).tolist()

                rewards.extend(logits)

        return rewards

    def set_context(self, context: List[str]):
        self.prompt_txt = context

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int] = None,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        if self.prompt_txt is None:
            context_candidates = candidates
        else:

            context_candidates = [
                prompt + str2 for prompt, str2 in zip(self.prompt_txt, candidates)
            ]

        return self._evaluate(context_candidates, **kwargs)
