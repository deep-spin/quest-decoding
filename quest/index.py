import math
from numpy.random import randint


class IndexDistribution:
    def __init__(self, dist):
        self.dist = dist

    def log_prob(self, index, truncation=None ):
        if truncation is not None:
            normalization = self.dist.cdf(truncation)
        else:
            normalization = 1
        return math.log(self.dist.pmf(index) / normalization)

    def sample(self, truncation=None):
        index = self.dist.rvs()
        while truncation is not None and index > truncation:
            index = self.dist.rvs()
        return index
    
class Uniform(IndexDistribution):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
        # self.mu = mu

    def sample(self, truncation):
        return randint(0, truncation)

    def log_prob(self, index, truncation):
        normalization = truncation

        return -math.log(normalization)

