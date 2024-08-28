from typing import List
import numpy as np


class Reward:
    """
    The base class for reward evaluation.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name.replace(
            "/", "-"
        ).split(".")[0]

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.

        """
        raise NotImplementedError


class ConstantReward(Reward):
    """
    A class for a constant reward.

    Attributes:
        reward (float): The reward value.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, reward: float):
        """
        The constructor for ConstantReward class.

        Args:
            reward (float): The reward value.

        """
        self.reward = reward
        super().__init__(
            f"constant:{self.reward}"
        )

    def evaluate(
        self,
        candidates: List[str],
        accepted_indices: List[int],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """

        if accepted_indices is None:
            accepted_indices = list(
                range(len(candidates))
            )

        return [
            self.reward
            for _ in accepted_indices
        ]

    def set_context(self, *args, **kwargs):
        pass


class BackwardReward(Reward):
    """
    A class for a reward based on a backward model.

    Attributes:
        model (Model): The backward model to use for reward evaluation.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, model: Reward):
        """
        The constructor for BackwardReward class.

        Args:
            model (Model): The backward model to use for reward evaluation.

        """
        self.model = model
        super().__init__(
            f"b:{self.model.get_name()}"
        )

    def evaluate(
        self,
        candidates: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """

        return [
            -s
            for s in self.model.evaluate(
                candidates, **kwargs
            )
        ]


class RewardMix(Reward):
    def __init__(
        self,
        rewards: List[Reward],
        mixing_weights: List[float] = None,
    ):
        self.rewards = rewards
        self.mixing_weights = (
            mixing_weights
            if mixing_weights is not None
            else [1.0] * len(rewards)
        )

        assert len(self.rewards) == len(
            self.mixing_weights
        ), "The number of rewards must match the number of mixing weights."

        super().__init__(
            f"mix:{','.join([r.get_name() for r in rewards])}"
        )

    def evaluate(
        self,
        candidates: List[str],
        **kwargs,
    ) -> List[float]:

        ## TODO THIS SHOULD BE DONE IN PARALLEL !!!!!!!
        evaluations = [
            r.evaluate(candidates, **kwargs)
            for r in self.rewards
        ]

        return (
            np.stack(
                [
                    np.array(e) * w
                    for e, w in zip(
                        evaluations,
                        self.mixing_weights,
                    )
                ],
                axis=0,
            )
            .sum(0)
            .tolist()
        )
