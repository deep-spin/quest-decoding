import numpy as np


def clamp_logit(score,clamp=1e-3):
    
    logit = lambda x: np.log(x / (1 - x)) 
    return logit(np.clip(score,clamp,1-clamp))