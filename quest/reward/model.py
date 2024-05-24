from typing import List, Union
from quest.reward.base import Reward

import numpy as np
from transformers import pipeline
import logging 
from quest.utils.logger import fix_loggers
from quest.utils.math import clamp_logit
# Ignore specific UserWarning from transformers library
# Configure the transformers logger

fix_loggers(name="transformers")
    

class RewardModel(Reward):
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

    def __init__(self, model_path: str, batch_size: int = 32, device:int = 0, task:str="text-classification", clamp:float=1e-3):
        super().__init__()
        
        self.batch_size = batch_size
        self.device = device
        self.sentiment_pipe = pipeline(task, model=model_path, device=device)
        self.sent_kwargs = {"batch_size": self.batch_size}
        self.clamp = clamp

    def evaluate(self, candidates: List[str], **kwargs) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """
        return [
            clamp_logit(sent["score"],self.clamp) 
            for sent in self.sentiment_pipe(candidates, **self.sent_kwargs) 
        ]
