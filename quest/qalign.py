from quest.core import Quest
from quest.model.base import LanguageModel
from quest.reward.base import Reward
from quest.proposal import RLHFSuffixProposal

from typing import List, Dict


class QAlign(Quest):
    """
    QAlign is a class that implements the Q-Learning algorithm for aligning text and images.
    It inherits from the Quest class and uses a language model and a reward function to perform
    the alignment task.
    """

    def __init__(
        self,
        input_data: List[Dict[str, str]],
        model: LanguageModel,
        reward: Reward,
        **kwargs
    ):
        """
        Initializes the QAlign class.
        Args:
            input_data (List[Dict[str, str]]): A list of dictionaries containing the input data.
            model (LanguageModel): The language model to be used for alignment.
            reward (Reward): The reward function to be used for the alignment task.
            **kwargs: Additional keyword arguments.
        """
        # Ensure input_data is a list of dictionaries
        super().__init__(
            input_data=input_data,
            proposal=RLHFSuffixProposal(
                model=model,
                reward=reward,
            ),
            **kwargs
        )
