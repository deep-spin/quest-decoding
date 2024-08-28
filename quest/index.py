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

    def log_prob(
        self, index, truncation=None
    ):

        raise NotImplementedError

    def sample(self, truncation=None):
        """
        Generates a random sample from the distribution.

        Args:
            truncation: An optional truncation value.

        Returns:
            A random sample from the distribution.

        """
        raise NotImplementedError


class Discretized(IndexDistribution):

    def __init__(self, block_size: int):
        self.block_size = block_size

    def sample(self, truncation):
        return (
            randint(
                0,
                truncation
                // self.block_size,
            )
            * self.block_size
        )

    def log_prob(
        self, index, truncation
    ) -> float:
        normalization = float(
            truncation // self.block_size
        )

        return -math.log(normalization)


class Uniform(Discretized):

    def __init__(self):
        super().__init__(1)


class Constant(IndexDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self, truncation):
        return self.value

    def log_prob(self, index, truncation):
        return 0


class Zero(Constant):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)
