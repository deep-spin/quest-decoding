
import copy
import dataclasses
from typing import List, Dict

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

from quest.index import IndexDistribution
from quest.model.base import LanguageModel
from quest.reward.base import Reward
from quest.utils import join_accepted_values
#from utils.generate import encode_starting_root
#from utils.perf import timing

#os.environ["WANDB_SILENT"] = "true"


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

        reward: List[float]  # The reward obtained at the current state
        transition_scores: List[
            List[float]
        ]  # The scores for transitioning from the current state to other states
        completion: List[List[float]]  # The completion tensor for the current state
        text: List[str]  # The completion text for the current state
        index: List[int]  # The index of the current state
        t: int = 0  # The current time step

        def to_json(self):
            """
            This function converts the State class to a JSON object.
            """
            return {
                "reward": self.reward,
                "transition_scores": self.transition_scores,
                "completion": self.completion,
                "text": self.text,
                "t": self.t,
                "index": self.index,
            }

        def copy_relevant(self, relevant_chains: List[int]):

            return Quest.State(
                reward=[self.reward[i] for i in relevant_chains],
                transition_scores=[
                    self.transition_scores[i] for i in relevant_chains
                ],
                completion=[self.completion[i] for i in relevant_chains],
                text=[self.text[i] for i in relevant_chains],
                t=self.t,
                index=[self.index[i] for i in relevant_chains],
            )

        def paste_relevant(self, relevant_chains: List[int], additions):

            new_state = copy.deepcopy(self)

            for i, chain in enumerate(relevant_chains):
                new_state.reward[chain] = additions.reward[i]
                new_state.transition_scores[chain] = additions.transition_scores[i]
                new_state.completion[chain] = additions.completion[i]
                new_state.text[chain] = additions.text[i]
                new_state.index[chain] = additions.index[i]

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
        state_path: List[Dict[str,str]]
        
        
    def __init__(
        self,
        input_data: List[Dict[str,str]],
        model: LanguageModel,
        dist: IndexDistribution,
        reward: Reward,
        beta: float = 0.1,
        temperature: float = 0.9,
        avoid_redundancy: bool = True,
        logratio_clamp=20,
    ):
        """
        Initializes the ARMarkovChain class.

        Parameters:
        model: The model to LLM used as the completion model.
        reward_model: The model to be used for calculating the reward.
        beta (float, optional): The beta value for the reward calculation. Default is 0.2.
        temperature (float, optional): The temperature for the LLM model. Default is 1.5.
        """
        self.chains = len(input_data)
        self.input_data = input_data
        self.rm = reward
        self.beta = beta
        self.model = model
        self.temperature = temperature
        self.dist = dist
        self.avoid_redundancy = avoid_redundancy
        self.state_path = []
        self.accepted_indices = [[] for _ in range(self.chains)]
        self.rejected_indices = [[] for _ in range(self.chains)]
        self.samples = [[] for _ in range(self.chains)]
        self.logratio_clamp =logratio_clamp

        self.steps = 0

    def compute_reward(self, proposal_text, uncomplete_indices=None):
        """
        This function calculates the log reward for a given proposal text.

        Parameters:
        proposal_text (str): The text for which the reward is to be calculated.

        Returns:
        float: The calculated log reward.

        The function works as follows:
        1. It first uses the reward model's forward method to calculate the reward for the proposal text.
        2. It then normalizes this reward by subtracting the maximum possible reward (as given by the reward model)
           and dividing by the beta value.
        3. The result is the log reward for the proposal text.
        """

        value = self.rm.evaluate(proposal_text, accepted_indices=uncomplete_indices)


        return [ v/self.beta for v in value ]

    def start_chain(self, prompt, warm_start=None) -> State:

        if warm_start is None:
            return self.draw_initial_state(prompt)
        else:
            return self.bootstrap_initial_sate(prompt, warm_start)

    def bootstrap_initial_sate(self, prompt, samples):

        self.samples = copy.deepcopy(samples)
        completions_text = [s[-1] for s in self.samples]  # list of samples.

        completions, transition_scores = self.model.evaluate_continuation(
            prompt, completions_text, temperature=self.temperature
        )

        reward = self.compute_reward(completions_text)

        # Create the initial state  armcmc.pyfor the Markov chain
        state = Quest.State(
            reward=reward,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        self.stack(state)

        return state

    def draw_initial_state(self, prompt) -> State:
        """
        This function initializes the Markov chain given a source sentence.

        Parameters:
        source_sentence (str): The source sentence to initialize the Markov chain.

        Returns:
        tuple: A tuple containing the initial state and the completion text.
        """

        # Tokenize the prompt from the source sentence

        # self.prompt_txt = [ self.model.get_prompt(ss) for ss in source_sentence ]
        # self.prompt = self.model.encode(source_sentence)
        # chains = len(prompt)

        # self.prompt = encode_starting_root(
        #    self.model.e,source_sentence
        # )

        # Generate the initial completion and transition scores
        completions, transition_scores = self.model.continuation(
            prompt,
            prefix=None,
            temperature=self.temperature,
        )

        # self.model.decode()

        # Decode the completion text
        completions_text = self.model.decode_tokenize(completions)

        # Compute the reward for the initial completion
        reward = self.compute_reward(completions_text)

        # Set the distribution for the index transition kernel
        # self.dist = [self.base_dist for completion in completions]

        # Create the initial state  armcmc.pyfor the Markov chain
        state = Quest.State(
            reward=reward,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        for i, t in enumerate(state.text):
            self.samples[i].append(t)

        # self.samples = [[t] for i, t in enumerate(state.text)]

        self.stack(state)
        # self.state_path = [{**state.to_json(), "accept": [True] * chains}]

        # logging.info(f"Initial state: {state.text}")

        return state

    def criterion(
        self, 
        previous_state, 
        proposal_state,
        indeces):
        
        previous_length =list(map(len,previous_state.completion))
        proposal_length = list(map(len,proposal_state.completion))
            
        index_log_likelihood_backward = np.array(
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

        index_log_likelihood_forward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(indeces, proposal_length)
            ],
        )

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
                for scores, index in zip(proposal_state.transition_scores, indeces)
            ]
        )

        log_likelihood_backward = (
            index_log_likelihood_backward + proposal_log_likelihood_backward
        )

        log_likelihood_forward = (
            index_log_likelihood_forward + proposal_log_likelihood_forward
        )

        log_transition_ratio = log_likelihood_backward - log_likelihood_forward

        log_reward_ratio = np.array(proposal_state.reward) - np.array(previous_state.reward)

        sum_value = log_reward_ratio + log_transition_ratio

        clamped_value = np.clip(sum_value, -self.logratio_clamp, self.logratio_clamp)
        detailed_balance = np.exp(clamped_value)
        
        alpha = (
                np.minimum(
                    detailed_balance,
                    np.ones_like(detailed_balance),
                )
            )
             
        return alpha
        
    def draw_transition(self, previous_state, prompt, uncomplete_indices=None):
        """
        This function performs one step of the Metropolis-Hastings MCMC algorithm.
        It generates a proposal state and calculates the detailed balance to decide whether to accept or reject the proposal.

        Parameters:
        previous_state (State): The previous state of the Markov Chain.

        Returns:
        tuple: A tuple containing the proposal state, the proposal text, and the detailed balance.
        """

        # _, t1 = previous_state.completion.shape
        indeces = [
            self.dist.sample(
                truncation=len(completion),
                #t=previous_state.t,
            )
            for completion in previous_state.completion
        ]

        # logging.debug(f"{'-'*40}\n Index: {index}")

        prefix = [
            completion[:index]
            for completion, index in zip(previous_state.completion, indeces)
        ]

        prefix_scores = [
            scores[:index]
            for scores, index in zip(previous_state.transition_scores, indeces)
        ]

        (
            continuation_proposal,
            continuation_transition_scores,
        ) = self.model.continuation(
            prompt,
            prefix,
            temperature=self.temperature,
        )  ## add mask here -

        proposal = list(
            map(
                lambda x: x[0] + x[1],
                zip(prefix, continuation_proposal),
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

        proposal_text = self.model.decode_tokenize(proposal)

        if self.avoid_redundancy:
            proposal_reward = self.compute_reward(
                proposal_text, uncomplete_indices=uncomplete_indices
            )
        else:
            proposal_reward = self.compute_reward(
                proposal_text,
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
            indeces
        )

        return (
            proposal_state,
            alpha,
        )

    def stack(self, state):
        self.state_path.append(
            {
                **state.to_json(),
                "sample_counts": [len(s) for s in self.samples],
            }
        )

    def get_index_of_uncompleted_chains(
        self,
    ):
        enough_accepts = lambda s: (len(s) - self.steps) >= 0
        inds = [i for i, s in enumerate(self.samples) if not enough_accepts(s)]
        # print(inds)
        return inds

    def get_index_of_completed_chains(
        self,
    ):
        enough_accepts = lambda s: (len(s) - self.steps) >= 0
        inds = [i for i, s in enumerate(self.samples) if enough_accepts(s)]
        return inds

    def run(
        self,
        steps=100,
        warm_start=None,
        use_tqdm=False,
        n=None,
    ):
        """
        This function runs the Markov Chain Monte Carlo (MCMC) method with Metropolis-Hastings algorithm.
        It iteratively draws transitions and decides whether to accept or reject them based on the detailed balance.

        Parameters:
        source_sentence (str): The source sentence to start the chain.
        steps (int): The number of steps to run the chain. Default is 100.
        clear_cache_steps (int): The number of steps after which to clear the cache. Default is 100.

        Returns:
        tuple: A tuple containing the samples and the fraction of rejections.
        """

        if n is None:
            n = steps

        self.steps = steps
        # wandb.init(project="tower-llm", entity="graf", name=run_name)
        # Draw the initial state
        self.prompt = self.model.encode(self.input_data)

        state = self.start_chain(
            self.prompt,
            # max_steps=steps,
            warm_start=warm_start,
        )

        prev_state = state

        # Initialize the samples and full lists with the initial state
        # chains = len(source_sentence)
        accepted = np.zeros(self.chains, dtype=np.int32)
        # self.samples = [[t] for t in state.text]
        # self.accepted_indices = [[] for _ in range(chains)]
        # self.rejected_indices = [[] for _ in range(chains)]
        # self.state_path = [{**state.to_json(), "accept": [True] * chains}]

        if use_tqdm:
            iter = tqdm(range(n))
        else:
            iter = range(n)
        # Run the chain for the specified number of steps
        for i in iter:
            # Draw a transition from the current state
            # print("--"*10)

            # check & filter the sate

            uncomplete_indices = self.get_index_of_uncompleted_chains()
            # complete_indices = self.get_index_of_uncompleted_chains()

            # print(f"uncomplete_chains: {len(uncomplete_indices)}")
            if len(uncomplete_indices) == 0:
                break

            if self.avoid_redundancy:
                prompt = [self.prompt[i] for i in uncomplete_indices]
                state = prev_state.copy_relevant(uncomplete_indices)

            else:
                prompt = self.prompt

            ## This may fail because of reward calculation!
            proposal_state, A = self.draw_transition(
                previous_state=state,
                prompt=prompt,
                uncomplete_indices=uncomplete_indices,
            )


            # Append the proposal to the full list
            # full.append((proposal_state.text[0], proposal_state.reward))

            ## logging.info([len(s) for s in self.samples])
            # Decide whether to accept the proposal
            accept = np.array(
                bernoulli(A).rvs(),
            ).reshape(A.shape)

            # accepted += accept

            # Define a function to decide the new state values based on acceptance

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
                text=join_accepted_values(accept, proposal_state.text, state.text),
                index=join_accepted_values(
                    accept,
                    proposal_state.index,
                    state.index,
                ),
                t=proposal_state.t,
            )

            if self.avoid_redundancy:
                prev_state = prev_state.paste_relevant(uncomplete_indices, state)
            else:
                prev_state = state

            chains_tochange = [
                i for i, predi in zip(uncomplete_indices, accept) if predi
            ]

            chains_notchange = [
                i for i, predi in zip(uncomplete_indices, accept) if not predi
            ]

            # chains_tochange = [i for i, predi in enumerate(accept) if predi]
            # chain_notchange = [i for i, predi in enumerate(accept) if not predi]

            accepted_indices_toadd = [
                indexi
                for predi, indexi in zip(accept, proposal_state.index)
                if predi
            ]
            samples_toadd = [
                texti for predi, texti in zip(accept, proposal_state.text) if predi
            ]

            rejected_indices_toadd = [
                indexi
                for predi, indexi in zip(accept, proposal_state.index)
                if not predi
            ]

            for index, chain in enumerate(chains_tochange):
                self.accepted_indices[chain].append(accepted_indices_toadd[index])
                self.samples[chain].append(samples_toadd[index])

            for index, chain in enumerate(chains_notchange):
                self.rejected_indices[chain].append(rejected_indices_toadd[index])

            self.stack(prev_state)


            # Clear the cache periodically

        # Calculate the fraction of rejections -
        #reject_fractions = accepted / steps

        return Quest.Output(
            samples=self.samples, 
            accepted_indices=self.accepted_indices, 
            rejected_indices=self.rejected_indices, 
            state_path=self.state_path)



class QuestRLHF(Quest):
    def __init__(self,**quest_kwargs):
        super().__init__(**quest_kwargs)
        

    def criterion(
        self, 
        previous_state, 
        proposal_state,
        indeces):
        
        previous_length =list(map(len,previous_state.completion))
        proposal_length = list(map(len,proposal_state.completion))
            
        index_log_likelihood_backward = np.array(
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

        index_log_likelihood_forward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(indeces, proposal_length)
            ],
        )

        log_likelihood_backward = (
            index_log_likelihood_backward 
        )

        log_likelihood_forward = (
            index_log_likelihood_forward 
        )

        log_transition_ratio = log_likelihood_backward - log_likelihood_forward

        log_reward_ratio = np.array(proposal_state.reward) - np.array(previous_state.reward)

        sum_value = log_reward_ratio + log_transition_ratio

        clamped_value = np.clip(sum_value, -self.logratio_clamp, self.logratio_clamp)
        detailed_balance = np.exp(clamped_value)
        
        alpha = (
                np.minimum(
                    detailed_balance,
                    np.ones_like(detailed_balance),
                )
            )
             
        return alpha
      
    
if __name__ == "__main__":
    import pdb

    pdb.set_trace()