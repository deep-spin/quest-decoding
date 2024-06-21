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



class ConstantReward(Reward):
    """
    A class for a constant reward.

    Attributes:
        reward (float): The reward value.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, reward:float):
        """
        The constructor for ConstantReward class.

        Args:
            reward (float): The reward value.

        """
        self.reward = reward

    def evaluate(self, candidates:List[str], accepted_indices:List[int],**kwargs)->List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """
        
        if accepted_indices is None:
            accepted_indices = list(range(len(candidates)))
        
        
        return [self.reward for _ in accepted_indices]



class BackwardReward(Reward):
    """
    A class for a reward based on a backward model.

    Attributes:
        model (Model): The backward model to use for reward evaluation.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, model:Reward):
        """
        The constructor for BackwardReward class.

        Args:
            model (Model): The backward model to use for reward evaluation.

        """
        self.model = model

    def evaluate(self, candidates:List[str],**kwargs)->List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """
        
        
        return [ -s for s in self.model.evaluate(candidates, **kwargs)]