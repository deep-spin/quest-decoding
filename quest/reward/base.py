from typing import List


class Reward:
    """
    The base class for reward evaluation.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def evaluate(self, candidates:List[str], accepted_indices:List[int],**kwargs)->List[float]:
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

