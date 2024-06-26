from typing import List
from quest.reward.base import Reward

from transformers import pipeline
from quest.utils.logger import fix_loggers
from quest.utils.math import clamp_logit
from quest.utils.data import get_loader
import numpy as np 
from tqdm import tqdm
# Ignore specific UserWarning from transformers library
# Configure the transformers logger

fix_loggers(name="transformers")

import torch
from torch.utils.data import Dataset, DataLoader

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
        clamp: float = 20,
    ):
        
        super().__init__(f"rm:{model_path}")
        

        self.batch_size = batch_size
        self.device = device
        self.model = pipeline(task, model=model_path, device=device)
        self.kwargs = {"batch_size": self.batch_size}
        self.clamp = clamp
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, candidates: List[str], accepted_indices: List[int]=None, use_tqdm=False, **kwargs) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int]): The list of indices of accepted candidates.
            batch_size (int, optional): The batch size for inference. Defaults to 32.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """
        
        if accepted_indices is None:
            accepted_indices = list(range(len(candidates)))
            
        candidates = [candidates[i] for i in accepted_indices]
        

        loader = get_loader(
            candidates,
            self.tokenizer,
            use_tqdm=use_tqdm,
            batch_size=self.batch_size
        )
        
        rewards = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs =self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = np.clip(outputs.logits[:,0].cpu().numpy(), -self.clamp, self.clamp).tolist()

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

    def __init__(
        self,
        **rm_kwargs
    ):
        super().__init__(**rm_kwargs)
        self.name="c"+self.name
        self.prompt_txt = None


    def set_context(self, context: List[str]):
        self.prompt_txt = context

    def evaluate(self, candidates: List[str], **kwargs) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """
     
        context_candidates = [str1 + str2 for str1, str2 in zip(self.prompt_txt, candidates)]
        
        return super().evaluate(context_candidates, **kwargs)
   
