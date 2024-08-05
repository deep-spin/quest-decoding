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
                # "completion": self.completion,
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
        accepted_indices: List[List[int]]
        rejected_indices: List[List[int]]
        state_path: List[Dict[str, str]]

    def __init__(
        self,
        input_data: List[Dict[str, str]],
        model: LanguageModel,
        reward: Reward,
        dist: IndexDistribution = Uniform(),
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
        self.model = model
        self.dist = dist
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
            return self.draw_initial_state(
                prompt
            )
        else:
            return (
                self.bootstrap_initial_sate(
                    prompt, warm_start
                )
            )

    def bootstrap_initial_sate(
        self, prompt, samples: List[str]
    ) -> State:
        """
        Bootstrap the initial state for the Markov chain. Setting specific samples as the start of the markov chain.

        Args:
            prompt (str): The prompt text.
            samples (list): List of samples.

        Returns:
            State: The initial state for the Markov chain.

        """
        self.samples = copy.deepcopy(
            samples
        )
        completions_text = [
            s[-1] for s in self.samples
        ]  # list of samples.

        completions, transition_scores = (
            self.model.evaluate_continuation(
                prompt,
                completions_text,
            )
        )

        reward = self.compute_reward(
            completions_text
        )

        # Create the initial state for the Markov chain
        state = Quest.State(
            reward=reward,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        self.stack(
            state,
            len(reward) * [1],
            len(reward) * [1.0],
        )

        return state

    def draw_initial_state(
        self, prompt
    ) -> State:
        """
        This function initializes the Markov chain given a prompt.

        Parameters:
        prompt (str): The prompt to initialize the Markov chain.

        Returns:
        State: The initial state of the Markov chain.

        Notes:
        - The Markov chain is initialized by generating the initial completion and transition scores.
        - The completion text is decoded from the generated completions.
        - The reward for the initial completion is computed.
        - The initial state for the Markov chain is created using the computed reward, transition scores,
          completions, completion text, and index.
        - The completion text is stored in the samples list for each index.
        - The initial state is added to the stack.

        """
        # Generate the initial completion and transition scores
        completions, transition_scores = (
            self.model.continuation(
                prompt,
                prefix=None,
            )
        )
        # Decode the completion text
        completions_text = (
            self.model.decode_tokenize(
                completions
            )
        )

        # Compute the reward for the initial completion
        reward = self.compute_reward(
            completions_text
        )

        # Create the initial state for the Markov chain
        state = Quest.State(
            reward=reward,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        for i, t in enumerate(state.text):
            self.samples[i].append(t)

        self.stack(
            state,
            len(reward) * [1],
            len(reward) * [1.0],
        )

        return state

    def criterion(
        self,
        previous_state: State,
        proposal_state: State,
        indeces: List[int],
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
        previous_length = list(
            map(
                len,
                previous_state.completion,
            )
        )
        proposal_length = list(
            map(
                len,
                proposal_state.completion,
            )
        )

        # Calculate the log likelihood ratios for the indices
        index_log_likelihood_forward = (
            np.array(
                [
                    self.dist.log_prob(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        indeces,
                        previous_length,
                    )
                ]
            )
        )

        index_log_likelihood_backward = (
            np.array(
                [
                    self.dist.log_prob(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        indeces,
                        proposal_length,
                    )
                ],
            )
        )

        # Calculate the log likelihood ratios for the rewards
        proposal_log_likelihood_backward = np.array(
            [
                np.sum(scores[index:])
                for scores, index in zip(
                    previous_state.transition_scores,
                    indeces,
                )
            ]
        )

        proposal_log_likelihood_forward = np.array(
            [
                np.sum(scores[index:])
                for scores, index in zip(
                    proposal_state.transition_scores,
                    indeces,
                )
            ]
        )

        # Calculate the log likelihood ratios for the indices and rewards
        log_likelihood_backward = (
            index_log_likelihood_backward
            + proposal_log_likelihood_backward
        )

        log_likelihood_forward = (
            index_log_likelihood_forward
            + proposal_log_likelihood_forward
        )

        # Calculate the log transition ratio
        log_transition_ratio = (
            log_likelihood_backward
            - log_likelihood_forward
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

        indeces = [
            self.dist.sample(
                truncation=len(completion),
                # t=previous_state.t,
            )
            for completion in previous_state.completion
        ]

        prefix = [
            completion[:index]
            for completion, index in zip(
                previous_state.completion,
                indeces,
            )
        ]

        prefix_scores = [
            scores[:index]
            for scores, index in zip(
                previous_state.transition_scores,
                indeces,
            )
        ]

        (
            continuation_proposal,
            continuation_transition_scores,
        ) = self.model.continuation(
            prompt,
            prefix,
        )  ## add mask here -

        proposal = list(
            map(
                lambda x: x[0] + x[1],
                zip(
                    prefix,
                    continuation_proposal,
                ),
            )
        )
        proposal_transition_scores = list(
            map(
                lambda x: x[0] + x[1],
                zip(
                    prefix_scores,
                    continuation_transition_scores,
                ),
            )
        )

        proposal_text = (
            self.model.decode_tokenize(
                proposal
            )
        )

        if self.avoid_redundancy:
            proposal_reward = self.compute_reward(
                proposal_text,
                uncomplete_indices=uncomplete_indices,
            )
        else:
            proposal_reward = (
                self.compute_reward(
                    proposal_text,
                )
            )

        proposal_state = Quest.State(
            completion=proposal,
            reward=proposal_reward,
            transition_scores=proposal_transition_scores,
            text=proposal_text,
            t=previous_state.t + 1,
            index=indeces,
        )

        alpha = self.criterion(
            previous_state,
            proposal_state,
            indeces,
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
        self.prompt = self.model.encode(
            self.input_data
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

            # Update the state values based on the acceptance
            state = Quest.State(
                completion=join_accepted_values(
                    accept,
                    proposal_state.completion,
                    state.completion,
                ),
                reward=join_accepted_values(
                    accept,
                    proposal_state.reward,
                    state.reward,
                ),
                transition_scores=join_accepted_values(
                    accept,
                    proposal_state.transition_scores,
                    state.transition_scores,
                ),
                text=join_accepted_values(
                    accept,
                    proposal_state.text,
                    state.text,
                ),
                index=join_accepted_values(
                    accept,
                    proposal_state.index,
                    state.index,
                ),
                t=proposal_state.t,
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

            chains_notchange = [
                i
                for i, predi in zip(
                    uncomplete_indices,
                    accept,
                )
                if not predi
            ]

            accepted_indices_toadd = [
                indexi
                for predi, indexi in zip(
                    accept,
                    proposal_state.index,
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

            rejected_indices_toadd = [
                indexi
                for predi, indexi in zip(
                    accept,
                    proposal_state.index,
                )
                if not predi
            ]

            for index, chain in enumerate(
                chains_tochange
            ):
                self.accepted_indices[
                    chain
                ].append(
                    accepted_indices_toadd[
                        index
                    ]
                )
                self.samples[chain].append(
                    samples_toadd[index]
                )

            for index, chain in enumerate(
                chains_notchange
            ):
                self.rejected_indices[
                    chain
                ].append(
                    rejected_indices_toadd[
                        index
                    ]
                )

            self.stack(
                prev_state,
                accept.tolist(),
                A.tolist(),
            )

        return Quest.Output(
            samples=self.samples,
            accepted_indices=self.accepted_indices,
            rejected_indices=self.rejected_indices,
            state_path=self.state_path,
        )


class QuestRLHF(Quest):
    def __init__(self, **quest_kwargs):
        super().__init__(**quest_kwargs)

    def criterion(
        self,
        previous_state,
        proposal_state,
        indeces,
    ):

        previous_length = list(
            map(
                len,
                previous_state.completion,
            )
        )
        proposal_length = list(
            map(
                len,
                proposal_state.completion,
            )
        )

        index_log_likelihood_forward = (
            np.array(
                [
                    self.dist.log_prob(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        indeces,
                        previous_length,
                    )
                ]
            )
        )

        index_log_likelihood_backward = (
            np.array(
                [
                    self.dist.log_prob(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        indeces,
                        proposal_length,
                    )
                ],
            )
        )

        log_likelihood_backward = (
            index_log_likelihood_backward
        )

        log_likelihood_forward = (
            index_log_likelihood_forward
        )

        log_transition_ratio = (
            log_likelihood_backward
            - log_likelihood_forward
        )

        log_reward_ratio = np.array(
            proposal_state.reward
        ) - np.array(previous_state.reward)

        sum_value = (
            log_reward_ratio
            + log_transition_ratio
        )

        clamped_value = np.clip(
            sum_value,
            -self.logratio_clamp,
            self.logratio_clamp,
        )
        detailed_balance = np.exp(
            clamped_value
        )

        alpha = np.minimum(
            detailed_balance,
            np.ones_like(detailed_balance),
        )

        return alpha


class QuestMetropolis(Quest):
    ## TODO: is implementation is clean but is really not efficient make this parallel.

    def __init__(self, **quest_kwargs):
        super().__init__(
            dist=Zero(), **quest_kwargs
        )


class QuestMetropolisRLHF(QuestRLHF):
    ## TODO: is implementation is clean but is really not efficient make this parallel.

    def __init__(self, **quest_kwargs):
        super().__init__(
            dist=Zero(), **quest_kwargs
        )
