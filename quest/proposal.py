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
from langchain.prompts import PromptTemplate
from quest.core import Quest


class LLMProposal(Quest.Proposal):

    def __init__(
        self,
        model: LanguageModel,
    ) -> None:
        super().__init__()

        self.model = model

    def draw_initial_state(
        self, prompt
    ) -> Quest.State:
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

        state = Quest.State(
            reward=None,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        return state

    def bootstrap_initial_state(
        self, prompt, samples: List[str]
    ) -> Quest.State:
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

        # Create the initial state for the Markov chain
        state = Quest.State(
            reward=None,
            transition_scores=transition_scores,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        return state

    def get_prompt(
        self,
        input_data: List[Dict[str, str]],
    ):
        return self.model.encode(input_data)

    def join_accepted_values(
        self,
        accept,
        previous_state: Quest.Proposal,
        proposal_state: Quest.Proposal,
    ):

        # Update the state values based on the acceptance
        state = Quest.State(
            completion=join_accepted_values(
                accept,
                proposal_state.completion,
                previous_state.completion,
            ),
            reward=join_accepted_values(
                accept,
                proposal_state.reward,
                previous_state.reward,
            ),
            transition_scores=join_accepted_values(
                accept,
                proposal_state.transition_scores,
                previous_state.transition_scores,
            ),
            text=join_accepted_values(
                accept,
                proposal_state.text,
                previous_state.text,
            ),
            index=(
                join_accepted_values(
                    accept,
                    proposal_state.index,
                    previous_state.index,
                )
                if (
                    proposal_state.index
                    is not None
                )
                else None
            ),
            t=proposal_state.t,
        )

        return state


class SuffixProposal(LLMProposal):

    def __init__(
        self,
        model: LanguageModel,
        dist: IndexDistribution = Uniform(),
    ):
        super().__init__(model=model)
        self.dist = dist

    def transition_likelihood_ratio(
        self,
        previous_state: Quest.State,
        proposal_state: Quest.State,
        **kwargs,
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

        # Calculate the log likelihood ratios for the indices
        index_log_likelihood_forward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(
                    proposal_state.index,
                    previous_length,
                )
            ]
        )

        index_log_likelihood_backward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(
                    proposal_state.index,
                    proposal_length,
                )
            ],
        )

        # Calculate the log likelihood ratios for the rewards
        proposal_log_likelihood_backward = np.array(
            [
                np.mean(scores[index:])
                for scores, index in zip(
                    previous_state.transition_scores,
                    proposal_state.index,
                )
            ]
        )

        proposal_log_likelihood_forward = np.array(
            [
                np.mean(scores[index:])
                for scores, index in zip(
                    proposal_state.transition_scores,
                    proposal_state.index,
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

        return log_transition_ratio

    def transition(
        self,
        previous_state: Quest.State,
        prompt: List[List[int]],
    ):

        completions = (
            previous_state.completion
        )

        indeces = [
            self.dist.sample(
                truncation=len(completion),
                # t=previous_state.t,
            )
            for completion in completions
        ]

        prefix = [
            completion[:index]
            for completion, index in zip(
                completions,
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
                proposal,
            )
        )

        proposal_state = Quest.State(
            completion=proposal,
            reward=None,
            transition_scores=proposal_transition_scores,
            text=proposal_text,
            index=indeces,
            t=previous_state.t + 1,
        )

        return proposal_state


class RLHFSuffixProposal(SuffixProposal):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def transition_likelihood_ratio(
        self,
        previous_state: Quest.State,
        proposal_state: Quest.State,
        **kwargs,
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

        # Calculate the log likelihood ratios for the indices
        index_log_likelihood_forward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(
                    proposal_state.index,
                    previous_length,
                )
            ]
        )

        index_log_likelihood_backward = np.array(
            [
                self.dist.log_prob(
                    index=index,
                    truncation=n,
                )
                for index, n in zip(
                    proposal_state.index,
                    proposal_length,
                )
            ],
        )

        # Calculate the log likelihood ratios for the indices and rewards
        log_likelihood_backward = (
            index_log_likelihood_backward
            # + proposal_log_likelihood_backward
        )

        log_likelihood_forward = (
            index_log_likelihood_forward
            # + proposal_log_likelihood_forward
        )

        # Calculate the log transition ratio
        log_transition_ratio = (
            log_likelihood_backward
            - log_likelihood_forward
        )

        return log_transition_ratio


BaseTransitionPrompt = PromptTemplate.from_template(
    "{prompt}Answer Given:{previous_answer}\nAlternative:{next_answer}"
)


class FancyProposal(LLMProposal):

    def __init__(
        self,
        transition_func,
        **kwargs,
    ):
        self.transition_func = (
            transition_func
        )

        super().__init__(**kwargs)

    def get_prompt(
        self,
        input_data: List[Dict[str, str]],
    ):

        prompt_txt = [
            self.model.get_prompt(**data)
            for data in input_data
        ]

        fstrings = [
            data["chat_template_prompt"]
            for data in input_data
        ]

        return (prompt_txt, fstrings)

    #     return prompt_txt

    def transition(
        self,
        previous_state: Quest.State,
        prompt: List[str],
    ):

        prompt, fstrings = prompt

        forward_context = [
            self.transition_func(
                chat_template_prompt=fstring,
                previous_answer=ptext.replace(
                    self.model.tokenizer.eos_token,
                    "",
                ),
                next_answer="",
                tokenizer=self.model.tokenizer,
            )
            for fstring, ptext in zip(
                fstrings,
                previous_state.text,
                # self.starting_output,
            )
        ]

        forward_context_tokens = (
            self.model.tokenize(
                forward_context
            )
        )

        (
            proposal,
            transition_scores,
        ) = self.model.continuation(
            forward_context_tokens,
            prefix=None,
        )

        proposal_text = (
            self.model.decode_tokenize(
                proposal
            )
        )

        proposal_state = Quest.State(
            completion=proposal,
            reward=None,
            transition_scores=transition_scores,
            text=proposal_text,
            index=None,
            t=previous_state.t + 1,
        )

        return proposal_state

    def transition_likelihood_ratio(
        self,
        previous_state: Quest.State,
        proposal_state: Quest.State,
        prompt: List[str],
    ):
        ## This has a serious error -> the transition is badlky computed.

        prompt, fstrings = prompt

        backward_context = [
            self.transition_func(
                chat_template_prompt=fstring,
                previous_answer=proposal,
                next_answer="",
                tokenizer=self.model.tokenizer,
            )
            for proposal, fstring in zip(
                proposal_state.text,
                fstrings,
            )
        ]

        backward_context_tokens = (
            self.model.tokenize(
                backward_context
            )
        )

        _, backward_transition_scores = (
            self.model.evaluate_continuation(
                backward_context_tokens,
                previous_state.text,
            )
        )

        log_likelihood_forward = np.array(
            list(
                map(
                    np.mean,
                    proposal_state.transition_scores,
                )
            )
        )

        log_likelihood_backward = np.array(
            list(
                map(
                    np.mean,
                    backward_transition_scores,
                )
            )
        )

        ## TODO I should do a log ratio of the LM base generation as well ..

        # Calculate the log transition ratio
        log_transition_ratio = (
            log_likelihood_backward
            - log_likelihood_forward
        )

        return log_transition_ratio

    def draw_initial_state(
        self, prompt
    ) -> Quest.State:

        prompt, _ = prompt

        state = super().draw_initial_state(
            self.model.tokenize(prompt)
        )

        self.starting_output = [
            o.split(" ")[0]
            for o in state.text
        ]

        return state

    def bootstrap_initial_state(
        self, prompt, samples: List[str]
    ) -> Quest.State:

        prompt, fstrings = prompt

        return (
            super().bootstrap_initial_state(
                self.mode.tokenize(prompt),
                samples,
            )
        )


class AlwaysAcceptFancyProposal(
    FancyProposal
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transition_likelihood_ratio(
        self,
        previous_state: Quest.State,
        proposal_state: Quest.State,
        prompt: List[str],
    ):

        prompt, fstrings = prompt

        return np.zeros(
            (len(prompt),), dtype=np.float32
        )
