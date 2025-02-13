from quest.reward.base import Reward
from quest.utils.math import clamp_logit
from comet import (
    download_model,
    load_from_checkpoint,
)
from typing import List
import os

from metricx24.models import MT5ForRegression
import transformers
import datasets
import torch
from typing import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CometModel(Reward):

    def __init__(
        self,
        model_path="Unbabel/XCOMET-XL",
        batch_size: int = 32,
        devices=[0],
        clamp: float = 1e-3,
        device_count=1,
        name="comet",
    ):
        super().__init__("qe:" + model_path)

        model_path = download_model(model_path)
        self.model = load_from_checkpoint(model_path, strict=False)
        self.batch_size = batch_size
        self.device_count = device_count
        self.devices = devices
        self.clamp = clamp
        self.sources = None
        self.references = None

    def set_context(self, context: List[Tuple[str, str]]):
        self.sources, self.references = zip(*context)
        self.sources = list(self.sources)
        self.references = list(self.references)

    def set_sources(self, sources: List[str]):
        self.sources = sources

    def set_references(self, references: List[str]):
        self.references = references

    def make_input(
        self,
        candidates: List[str],
        accepted_indices: List[int],
    ):

        data = [
            {
                "src": self.sources[i],
                "mt": c,
                "ref": self.references[i],
            }
            for c, i in zip(candidates, accepted_indices)
        ]

        return data

    def evaluate(
        self, candidates: List[str], accepted_indices=None, **kwargs
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        if accepted_indices is None:
            accepted_indices = list(range(len(candidates)))

        data = self.make_input(candidates, accepted_indices)

        return [
            clamp_logit(score, self.clamp)
            for score in self.model.predict(
                data,
                batch_size=self.batch_size,
                gpus=self.device_count,
                devices=self.devices,
                length_batching=False,
                num_workers=1,
            )["scores"]
        ]


class QEModel(CometModel):

    def __init__(self, model_path="Unbabel/wmt22-cometkiwi-da", **kwargs):
        super().__init__(model_path=model_path, **kwargs)

    def set_context(self, context: List[str]):
        self.sources = context

    def make_input(
        self,
        candidates: List[str],
        accepted_indices: List[int],
    ):
        return [
            {
                "src": self.sources[i],
                "mt": c,
            }
            for c, i in zip(candidates, accepted_indices)
        ]


class CustomDataCollator:
    def __init__(self, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract inputs (e.g., "input_ids") and other fields
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        # Pad input_ids and attention_masks to the same length
        padded_inputs = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            },
            padding="longest",  # Pad to the longest sequence in the batch
            max_length=self.max_length,  # Optional: truncate to a max length
            return_tensors="pt",
        )

        return padded_inputs


class MetricXModel(Reward):

    def __init__(
        self,
        model_path="google/metricx-23-xl-v2p0",  # google/metricx-24-hybrid-xl-v2p6
        tokenizer_name="google/mt5-xl",
        max_input_length: int = 1200,
        batch_size: int = 8,
        devices=[0],
        device_count=1,
    ):

        super().__init__("mt:" + model_path)

        self.device = torch.device(f"cuda:{devices[0]}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = MT5ForRegression.from_pretrained(model_path, torch_dtype="auto")
        if device_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.sources = None
        self.references = None

    def set_sources(self, sources: List[str]):
        self.sources = sources

    def set_references(self, references: List[str]):
        self.references = references

    def make_input(
        self, candidates: List[str], accepted_indices: List[int], is_qe=False
    ):
        data = [
            {
                "src": self.sources[i],
                "mt": c,
                "ref": self.references[i],
            }
            for c, i in zip(candidates, accepted_indices)
        ]

        dataset = datasets.Dataset.from_list(data)
        dataset = dataset.map(
            lambda example: self._make_input(example, is_qe),
        )
        dataset = dataset.map(
            self._tokenize,
        )
        dataset = dataset.map(self._remove_eos)
        """dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )"""
        return dataset

    def _make_input(self, example, is_qe):
        if is_qe:
            example["input"] = (
                "source: " + example["src"] + " candidate: " + example["mt"]
            )
        else:
            example["input"] = (
                "source: "
                + example["src"]
                + " candidate: "
                + example["mt"]
                + " reference: "
                + example["ref"]
            )
        return example

    def _tokenize(self, example):
        return self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding=False,  # "max_length",  # "longest",
        )

    def _remove_eos(self, example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def evaluate(
        self, candidates: List[str], accepted_indices=None, is_qe=False, **kwargs
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int], optional): Indices corresponding to the candidates.
            is_qe (bool, optional): Whether to use QE input format. Default is False.

        Returns:
            List[float]: The list of reward values for each candidate sequence.
        """
        if accepted_indices is None:
            accepted_indices = list(range(len(candidates)))

        dataset = self.make_input(candidates, accepted_indices, is_qe)

        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer, max_length=self.max_input_length
        )

        trainer = transformers.Trainer(
            model=self.model,
            data_collator=data_collator,
            args=transformers.TrainingArguments(
                per_device_eval_batch_size=self.batch_size,
                output_dir="./results",
                remove_unused_columns=False,
                disable_tqdm=False,
                # padding_strategy="longest",
            ),
        )
        predictions, _, _ = trainer.predict(
            test_dataset=dataset,
        )
        return (predictions / 25).tolist()
