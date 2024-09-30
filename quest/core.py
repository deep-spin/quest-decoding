import copy
import dataclasses
from typing import *

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

from quest.index import (
    IndexDistribution,
    Zero,
    Uniform,
)
from quest.model.base import LanguageModel
from quest.reward.base import Reward
from quest.utils.list import (
    join_accepted_values,
)


class Quest:
    """
    This class implements the Metropolis-Hastings MCMC method with an AR transition kernel.
    The Metropolis-Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random
    samples from a probability distribution for which direct sampling is difficult. This sequence can be used to
    approximate the distribution.

    The transition kernel works the following way:
    1. Sample a token index from 0 to the sentence legth.
    2. Sample a continuation from the current state until the eos token
    """

    @dataclasses.dataclass
    class State:
        """
        This class represents the state of the Markov Chain.
        """

        reward: List[
            float
        ]  # The reward obtained at the current state
        transition_scores: List[
            List[float]
        ]  # The scores for transitioning from the current state to other states
        completion: List[
            List[float]
        ]  # The completion tensor for the current state
        text: List[
            str
        ]  # The completion text for the current state
        index: List[
            int
        ]  # The index of the current state
        t: int = 0  # The current time step

        def to_json(self):
            """
            This function converts the State class to a JSON object.
            """
            return {
                "reward": self.reward,
                # "transition_scores": self.transition_scores,
                "completion": self.completion,
                "text": self.text,
                "t": self.t,
                "index": self.index,
            }

        def copy_relevant(
            self, relevant_chains: List[int]
        ):

            return Quest.State(
                reward=[
                    self.reward[i]
                    for i in relevant_chains
                ],
                transition_scores=[
                    self.transition_scores[
                        i
                    ]
                    for i in relevant_chains
                ],
                completion=[
                    self.completion[i]
                    for i in relevant_chains
                ],
                text=[
                    self.text[i]
                    for i in relevant_chains
                ],
                t=self.t,
                index=[
                    self.index[i]
                    for i in relevant_chains
                ],
            )

        def paste_relevant(
            self,
            relevant_chains: List[int],
            additions,
        ):

            new_state = copy.deepcopy(self)

            for i, chain in enumerate(
                relevant_chains
            ):
                new_state.reward[chain] = (
                    additions.reward[i]
                )
                new_state.transition_scores[
                    chain
                ] = additions.transition_scores[
                    i
                ]
                new_state.completion[
                    chain
                ] = additions.completion[i]
                new_state.text[chain] = (
                    additions.text[i]
                )
                new_state.index[chain] = (
                    additions.index[i]
                )

            new_state.t = additions.t

            return new_state

    @dataclasses.dataclass
    class Output:
        """
        This class represents the output of the Markov Chain.
        """

        samples: List[str]
        # accepted_indices: List[List[int]]
        # rejected_indices: List[List[int]]
        state_path: List[Dict[str, str]]

    class Proposal:

        def transition(
            self,
            previous_state,
            prompt: List[List[int]],
        ):
            raise NotImplementedError

        def transition_likelihood_ratio(
            self,
            previous_state,
            proposal_state,
            **kwargs
        ):
            raise NotImplementedError

        def get_prompt(
            self,
            input_data: List[
                Dict[str, str]
            ],
        ):
            raise NotImplementedError

        def bootstrap_initial_state(
            self, prompt, samples: List[str]
        ):
            raise NotImplementedError

        def draw_initial_state(
            self, prompt
        ):
            raise NotImplementedError

        def join_accepted_values(
            self,
            accept,
            previous_state,
            proposal_state,
        ):
            raise NotImplementedError

    def __init__(
        self,
        input_data: List[Dict[str, str]],
        proposal: Proposal,
        reward: Reward,
        beta: float = 0.1,
        avoid_redundancy: bool = True,
        logratio_clamp: float = 20,
    ):
        """
        Initializes the Quest class.

        Parameters:
        - input_data (List[Dict[str, str]]): The input data for the Quest class.
        - model (LanguageModel): The language model to be used as the completion model.
        - dist (IndexDistribution): The index distribution.
        - reward (Reward): The reward model to be used for calculating the reward.
        - beta (float, optional): The beta value for the reward calculation. Default is 0.1.
        - avoid_redundancy (bool, optional): Whether to avoid redundancy in the generated samples. Default is True.
        - logratio_clamp (int, optional): The maximum value for the log ratio. Default is 20.
        """
        self.chains = len(input_data)
        self.input_data = input_data
        self.rm = reward
        self.beta = beta
        self.proposal = proposal
        self.avoid_redundancy = (
            avoid_redundancy
        )
        self.state_path = []
        self.accepted_indices = [
            [] for _ in range(self.chains)
        ]
        self.rejected_indices = [
            [] for _ in range(self.chains)
        ]
        self.samples = [
            [] for _ in range(self.chains)
        ]
        self.logratio_clamp = logratio_clamp

        self.steps = 0

    def compute_reward(
        self,
        proposal_text: List[str],
        uncomplete_indices: Union[
            None, List[int]
        ] = None,
    ) -> List[float]:
        """
        This function calculates the log reward for a given proposal text.

        Parameters:
        proposal_text (str): The text for which the reward is to be calculated.

        Returns:
        float: The calculated log reward.

        """

        value = self.rm.evaluate(
            proposal_text,
            accepted_indices=uncomplete_indices,
        )

        return [
            v / self.beta for v in value
        ]

    def start_chain(
        self, prompt, warm_start=None
    ) -> State:

        if warm_start is None:

            state = self.proposal.draw_initial_state(
                prompt
            )

            # Compute the reward for the initial completion

            for i, t in enumerate(
                state.text
            ):
                self.samples[i].append(t)

        else:
            self.samples = copy.deepcopy(
                warm_start
            )

            state = self.proposal.bootstrap_initial_state(
                prompt,
                warm_start,
            )

        # Compute the reward for the initial completion

        state.reward = self.compute_reward(
            state.text
        )

        self.stack(
            state,
            len(state.reward) * [1],
            len(state.reward) * [1.0],
        )

        return state

    def criterion(
        self,
        previous_state: State,
        proposal_state: State,
        **kwargs
    ) -> np.ndarray:
        """
        This function calculates the Metropolis-Hastings criterion for accepting or rejecting a proposal state.

        Parameters:
        previous_state (State): The previous state of the Markov Chain.
        proposal_state (State): The proposal state of the Markov Chain.
        indeces (List[int]): The indices of the proposal state.

        Returns:
        np.ndarray: The acceptance probabilities for each proposal.

        Notes:
        - The Metropolis-Hastings criterion is calculated based on the log likelihood ratios of the indices and rewards.
        - The log likelihood ratios are clamped to avoid numerical instability.
        - The detailed balance is calculated as the exponential of the sum of the log likelihood ratios.
        - The acceptance probabilities are calculated as the minimum of the detailed balance and 1.

        """

        log_transition_ratio = self.proposal.transition_likelihood_ratio(
            previous_state=previous_state,
            proposal_state=proposal_state,
            **kwargs
        )

        # Calculate the log reward ratio
        log_reward_ratio = np.array(
            proposal_state.reward
        ) - np.array(previous_state.reward)

        # Calculate the sum of the log transition ratio and log reward ratio
        sum_value = (
            log_reward_ratio
            + log_transition_ratio
        )

        # Clamp the sum value to avoid numerical instability
        clamped_value = np.clip(
            sum_value,
            -self.logratio_clamp,
            self.logratio_clamp,
        )

        # Calculate the detailed balance as the exponential of the clamped sum value
        detailed_balance = np.exp(
            clamped_value
        )

        # Calculate the acceptance probabilities as the minimum of the detailed balance and 1
        alpha = np.minimum(
            detailed_balance,
            np.ones_like(detailed_balance),
        )
        return alpha

    def draw_transition(
        self,
        previous_state: State,
        prompt,
        uncomplete_indices: Union[
            None, List[int]
        ] = None,
    ) -> Tuple[State, np.ndarray]:
        """
        This function performs one step of the Metropolis-Hastings MCMC algorithm.
        It generates a proposal state and calculates the detailed balance to decide whether to accept or reject the proposal.

        Parameters:
        previous_state (State): The previous state of the Markov Chain.

        Returns:
        tuple: A tuple containing the proposal state, the proposal text, and the detailed balance.
        """

        proposal_state = (
            self.proposal.transition(
                previous_state, prompt
            )
        )

        if self.avoid_redundancy:
            proposal_reward = self.compute_reward(
                proposal_state.text,
                uncomplete_indices=uncomplete_indices,
            )
        else:
            proposal_reward = (
                self.compute_reward(
                    proposal_state.text,
                )
            )

        proposal_state.reward = (
            proposal_reward
        )

        alpha = self.criterion(
            previous_state,
            proposal_state,
            prompt=prompt,
        )

        return (
            proposal_state,
            alpha,
        )

    def stack(
        self,
        state: State,
        accept: List[bool],
        alpha: List[float],
    ):

        self.state_path.append(
            {
                **state.to_json(),
                "sample_counts": [
                    len(s)
                    for s in self.samples
                ],
                "accept": accept,
                "criterion": alpha,
            }
        )

    def get_index_of_uncompleted_chains(
        self,
    ) -> List[int]:
        enough_accepts = (
            lambda s: (len(s) - self.steps)
            >= 0
        )
        inds = [
            i
            for i, s in enumerate(
                self.samples
            )
            if not enough_accepts(s)
        ]
        return inds

    def get_index_of_completed_chains(
        self,
    ) -> List[int]:
        enough_accepts = (
            lambda s: (len(s) - self.steps)
            >= 0
        )
        inds = [
            i
            for i, s in enumerate(
                self.samples
            )
            if enough_accepts(s)
        ]
        return inds

    def run(
        self,
        steps: int = 100,
        warm_start: Union[
            None, List[str]
        ] = None,
        use_tqdm: bool = False,
        n: Union[None, int] = None,
    ) -> Output:
        """
        This function runs the Markov Chain Monte Carlo (MCMC) method with Metropolis-Hastings algorithm.
        It iteratively draws transitions and decides whether to accept or reject them based on the detailed balance.

        Parameters:
        - steps (int): The number of steps to run the chain. Default is 100.
        - warm_start (Union[None, List[str]]): A list of warm start sentences to initialize the chain. Default is None.
        - use_tqdm (bool): Whether to use tqdm for progress bar. Default is False.
        - n (Union[None, int]): The number of iterations to run the chain. Default is None, which is equal to the number of steps.

        Returns:
        - Output: A named tuple containing the samples, accepted indices, rejected indices, and state path.
        """

        if n is None:
            n = steps

        self.steps = steps
        # Draw the initial state
        self.prompt = (
            self.proposal.get_prompt(
                self.input_data
            )
        )

        state = self.start_chain(
            self.prompt,
            warm_start=warm_start,
        )

        prev_state = state

        if use_tqdm:
            iter = tqdm(range(n))
        else:
            iter = range(n)
        # Run the chain for the specified number of steps
        for i in iter:

            uncomplete_indices = (
                self.get_index_of_uncompleted_chains()
            )

            if len(uncomplete_indices) == 0:
                break

            if self.avoid_redundancy:
                prompt = [
                    self.prompt[i]
                    for i in uncomplete_indices
                ]
                state = prev_state.copy_relevant(
                    uncomplete_indices
                )

            else:
                prompt = self.prompt

            proposal_state, A = (
                self.draw_transition(
                    previous_state=state,
                    prompt=prompt,
                    uncomplete_indices=uncomplete_indices,
                )
            )

            # Decide whether to accept the proposal
            accept = np.array(
                bernoulli(A).rvs(),
            ).reshape(A.shape)

            state = self.proposal.join_accepted_values(
                accept=accept,
                previous_state=state,
                proposal_state=proposal_state,
            )

            self.stack(
                proposal_state,
                accept.tolist(),
                A.tolist(),
            )

            if self.avoid_redundancy:
                prev_state = prev_state.paste_relevant(
                    uncomplete_indices,
                    state,
                )
            else:
                prev_state = state

            chains_tochange = [
                i
                for i, predi in zip(
                    uncomplete_indices,
                    accept,
                )
                if predi
            ]

            samples_toadd = [
                texti
                for predi, texti in zip(
                    accept,
                    proposal_state.text,
                )
                if predi
            ]

            for index, chain in enumerate(
                chains_tochange
            ):

                self.samples[chain].append(
                    samples_toadd[index]
                )

        return Quest.Output(
            samples=self.samples,
            # accepted_indices=self.accepted_indices,
            # rejected_indices=self.rejected_indices,
            state_path=self.state_path,
        )
