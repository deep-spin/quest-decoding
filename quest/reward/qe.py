from quest.reward.base import Reward
from quest.utils.math import clamp_logit
from comet import download_model, load_from_checkpoint
from typing import List
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QEModel(Reward):
    # translation quality estimation

    def __init__(
        self,
        model_path="Unbabel/wmt23-cometkiwi-da-xl",
        batch_size: int = 32,
        device_count=1,
        clamp: float = 1e-3,
    ):
        super().__init__()

        model_path = download_model(model_path)
        self.model = load_from_checkpoint(model_path)
        self.batch_size = batch_size
        self.device_count = device_count
        self.clamp = clamp
        self.sources = None

    def set_sources(self, sources: List[str]):
        self.sources = sources

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

        assert (
            self.sources is not None
        ), "Please set sources before evaluating candidates."

        data = [{"src": self.sources[i], "mt": candidates[i]} for i in accepted_indices]

        return [
            clamp_logit(score, self.clamp)
            for score in self.model.predict(
                data, batch_size=self.batch_size, gpus=self.device_count
            )["scores"]
        ]
