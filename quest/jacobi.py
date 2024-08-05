from typing import *


import numpy as np

from quest.model.hf import HF
from quest.proposal import LLMProposal
from quest.reward.base import ConstantReward
from quest.core import Quest

from quest.utils.list import (
    join_accepted_values,
)


class JacobiProposal(LLMProposal):

    def __init__(
        self,
        model: HF,
    ):
        super().__init__(model=model)

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

        (
            self.prompt_key_values,
            self.prompt_logits,
            self.prompt_attention_mask,
        ) = self.model.get_starting_cache(
            prompt
        )

        generated, generated_logits = (
            self.model.ancestral(
                prompt=prompt,
            )
        )

        (
            generated,
            generated_logits,
            scores,
        ) = self.model.jacobi(
            response=generated,
            prev_logits=generated_logits,
            prompt_logits=self.prompt_logits,
            prompt_attention_mask=self.prompt_attention_mask,
            prompt_key_values=self.prompt_key_values,
        )

        state = Quest.State(
            reward=None,
            transition_scores={
                "scores": scores,
                "logits": generated_logits,
            },
            completion=generated,
            text=self.model.decode_tokenize(
                generated
            ),
            index=None,
        )

        return state

    def transition_likelihood_ratio(
        self,
        previous_state: Quest.State,
        proposal_state: Quest.State,
        **kwargs,
    ):

        scores = proposal_state.transition_scores[
            "scores"
        ]

        target_ratio = np.sum(
            scores["next_lm_likelihood"],
            axis=-1,
        ) - np.sum(
            scores["prev_lm_likelihood"],
            axis=-1,
        )

        transition_ratio = np.sum(
            scores[
                "backward_transition_likelihood"
            ],
            axis=-1,
        ) - np.sum(
            scores[
                "forward_transition_likelihood"
            ],
            axis=-1,
        )

        return (
            target_ratio + transition_ratio
        )

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
            transition_scores={
                "scores": join_accepted_values(
                    accept,
                    proposal_state.transition_scores[
                        "scores"
                    ],
                    previous_state.transition_scores[
                        "scores"
                    ],
                ),
                "logits": join_accepted_values(
                    accept,
                    proposal_state.transition_scores[
                        "logits"
                    ],
                    previous_state.transition_scores[
                        "logits"
                    ],
                ),
            },
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

    def transition(
        self,
        previous_state: Quest.State,
        prompt: List[List[int]],
    ):

        (
            generated,
            generated_logits,
            scores,
        ) = self.model.jacobi(
            response=previous_state.completion,
            prev_logits=previous_state.transition_scores[
                "logits"
            ],
            prompt_logits=self.prompt_logits,
            prompt_attention_mask=self.prompt_attention_mask,
            prompt_key_values=self.prompt_key_values,
        )

        proposal_state = Quest.State(
            reward=None,
            transition_scores={
                "scores": scores,
                "logits": generated_logits,
            },
            completion=generated,
            text=self.model.decode_tokenize(
                generated
            ),
            index=None,
        )

        return proposal_state


if __name__ == "__main__":

    import os
    from datasets import load_dataset
    from qflow.utils.data import (
        processhh_data,
    )

    dataset_path = "Anthropic/hh-rlhf"
    temperature = 1.0
    gpu_memory_utilization = 0.8
    model_path = "allenai/tulu-2-7b"  # "openai-community/gpt2"  #  "allenai/tulu-2-7b"
    n = 1

    ds = load_dataset(
        dataset_path, split="test"
    )

    psd = ds.map(processhh_data)

    data_iterable = list(psd)[:n]

    model = HF(
        model_path=model_path,
        download_dir=os.environ.get(
            "HF_HOME", "/tmp/"
        ),
        temperature=temperature,
        dtype="bfloat16",
        max_new_tokens=90,
    )

    proposal = JacobiProposal(model=model)

    reward = ConstantReward(0.0)

    chain = Quest(
        input_data=data_iterable,
        reward=reward,
        proposal=proposal,
        beta=1.0,
        avoid_redundancy=False,
    )

    chain_outputs = chain.run(
        steps=10000,
        use_tqdm=True,
    )

    import pdb

    pdb.set_trace()

# print("passed all tests")
