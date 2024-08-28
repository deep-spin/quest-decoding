from quest.reward.base import Reward
from quest.utils.math import clamp_logit
from comet import (
    download_model,
    load_from_checkpoint,
)
from typing import List
import os

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"
)


class CometModel(Reward):

    def __init__(
        self,
        model_path="Unbabel/XCOMET-XL",
        batch_size: int = 32,
        devices=[0],
        clamp: float = 1e-3,
        name="comet",
    ):
        super().__init__("qe:" + model_path)

        model_path = download_model(
            model_path
        )
        self.model = load_from_checkpoint(
            model_path, strict=False
        )
        self.batch_size = batch_size
        self.device_count = len(devices)
        self.devices = devices
        self.clamp = clamp
        self.sources = None
        self.references = None

    def set_sources(
        self, sources: List[str]
    ):
        self.sources = sources

    def set_references(
        self, references: List[str]
    ):
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
            for c, i in zip(
                candidates, accepted_indices
            )
        ]

        return data

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices=None,
        **kwargs
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        if accepted_indices is None:
            accepted_indices = list(
                range(len(candidates))
            )

        data = self.make_input(
            candidates, accepted_indices
        )

        return [
            clamp_logit(score, self.clamp)
            for score in self.model.predict(
                data,
                batch_size=self.batch_size,
                gpus=self.device_count,
                devices=self.devices,
            )["scores"]
        ]


class QEModel(CometModel):

    def __init__(
        self,
        model_path="Unbabel/wmt22-cometkiwi-da",
        **kwargs
    ):
        super().__init__(
            model_path=model_path, **kwargs
        )

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
            for c, i in zip(
                candidates, accepted_indices
            )
        ]
