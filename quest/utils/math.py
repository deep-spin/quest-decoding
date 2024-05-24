import numpy as np


def clamp_logit(score, clamp=1e-3):
    """
    Applies the clamp and logit transformation to the given score.

    Parameters:
    score (float): The input score to be transformed.
    clamp (float, optional): The lower and upper bounds for the score. Defaults to 1e-3.

    Returns:
    float: The transformed score.

    """
    logit = lambda x: np.log(x / (1 - x)) 
    return logit(np.clip(score, clamp, 1 - clamp))
