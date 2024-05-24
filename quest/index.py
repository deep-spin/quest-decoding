import math
from numpy.random import randint


import math

class IndexDistribution:
    """
    Represents a distribution of indices.

    Args:
        dist: A probability distribution object.

    Attributes:
        dist: The probability distribution object.

    Methods:
        log_prob(index, truncation=None): Calculates the logarithm of the probability of an index.
        sample(truncation=None): Generates a random sample from the distribution.

    """

    def __init__(self, dist):
        self.dist = dist

    def log_prob(self, index, truncation=None):
        """
        Calculates the logarithm of the probability of an index.

        Args:
            index: The index for which to calculate the probability.
            truncation: An optional truncation value.

        Returns:
            The logarithm of the probability of the index.

        """
        if truncation is not None:
            normalization = self.dist.cdf(truncation)
        else:
            normalization = 1
        return math.log(self.dist.pmf(index) / normalization)

    def sample(self, truncation=None):
        """
        Generates a random sample from the distribution.

        Args:
            truncation: An optional truncation value.

        Returns:
            A random sample from the distribution.

        """
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

