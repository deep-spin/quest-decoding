import torch
from transformers import (
    AutoModelForCausalLM,
)

from transformers import (
    AutoModelForCausalLM,
)
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import os
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
)
from langchain.prompts import PromptTemplate
from torch.nn.functional import (
    log_softmax,
    softmax,
)

from quest.model.base import (
    LocalLanguageModel,
)
import numpy as np

DEFAULT_TEMPLATE = (
    PromptTemplate.from_template("{prompt}")
)


def sample_jacobi(
    logits,
    response_length,
    temperature=1.0,
    eos_token_id=2,
):
    samples = (
        torch.multinomial(
            (logits / temperature)
            .softmax(-1)
            .reshape(
                -1,
                logits.shape[-1],
            ),
            num_samples=1,
        )
        .reshape(
            (
                logits.shape[0],
                -1,
            )
        )
        .tolist()
    )

    new_response = []
    for sample, rn in zip(
        samples,
        response_length,
    ):
        (eos_indices,) = torch.nonzero(
            torch.tensor(sample)
            == eos_token_id,
            as_tuple=True,
        )

        eos_indices = eos_indices.tolist()
        stop = (
            eos_indices[0]
            if len(eos_indices) > 0
            else rn
        )
        new_response.append(
            sample[: stop + 1]
        )

    return new_response


def transition_scores(
    logprobs,
    response,
):
    transition_scores = [
        [
            lpi[ri].item()
            for lpi, ri in zip(
                lp[: len(r)],
                r,
            )
        ]
        for lp, r in zip(
            logprobs,
            response,
        )
    ]

    return transition_scores


def jacobi_joint_probability(
    prompt_logits,
    x_logits,
    x,
    y,
    temperature=1.0,
):

    total_logits = torch.cat(
        [
            prompt_logits[:, -1:],
            x_logits,
        ],
        dim=-2,
    )

    marginal_likelihood = transition_scores(
        logprobs=log_softmax(
            total_logits,
            dim=-1,
        ),
        response=x,
    )

    transition_likelihood = (
        transition_scores(
            logprobs=log_softmax(
                total_logits / temperature,
                dim=-1,
            ),
            response=y,
        )
    )

    return (
        (marginal_likelihood),
        (transition_likelihood),
    )


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(
        self,
        stops=[],
    ):
        StoppingCriteria.__init__(self),

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        stops=[],
    ):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]


class HF(LocalLanguageModel):

    def __init__(
        self,
        model_path: str,
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        temperature=1.0,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            prompt_template=prompt_template,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            stop_tokens=stop_tokens,
        )
        self.device_map = (
            f"cuda:{device_ids[0]}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            pad_token_id=self.tokenizer.pad_token_id,
            cache_dir=download_dir,
            # use_flash_attention_2=True,
            # load_in_8bit=True
        )
        # self.model.to_bettertransformer()

        # self.tokenizer = LlamaTokenizer.from_pretrained(
        #    self.model_path,
        #    padding_side="left",
        # )

        self.model.eval()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # allow tf32 on matmul
        )
        torch.backends.cudnn.allow_tf32 = (
            True  # allow tf32 on cudnn
        )

        # self.model = torch.compile(
        #    self.model
        # )

        self.stop_words_ids = [
            self.tokenizer.encode(
                stop_word,
                add_prefix_space=False,
                add_special_tokens=False,
            )  # by default the tokenizer is sending spaces. We don't want that
            for stop_word in stop_tokens  # , "[/INST]", "[INST]"
        ]

    @torch.no_grad()
    def test_cache_system(
        self,
    ):

        return None

    @torch.no_grad()
    def jacobi(
        self,
        response,
        prev_logits,
        prompt_logits,
        prompt_attention_mask,
        prompt_key_values,
    ):

        ## pad logits
        maxT = max(map(len, prev_logits))

        padded_logits = torch.zeros(
            (
                len(prev_logits),
                maxT,
                len(prev_logits[0][0]),
            )
        )
        ## fill the values
        for i, logits in enumerate(
            prev_logits
        ):
            padded_logits[
                i, : len(logits)
            ] = torch.tensor(logits)

        prev_logits = padded_logits.to(
            self.device_map
        )

        if (
            prev_logits.shape[1]
            > self.max_new_tokens - 1
        ):
            prev_logits = prev_logits[
                :,
                : self.max_new_tokens - 1,
            ]

            rlength = [
                min(
                    len(r),
                    self.max_new_tokens,
                )
                for r in response
            ]

        else:
            rlength = [
                len(r) for r in response
            ]

        new_response = sample_jacobi(
            logits=torch.cat(
                [
                    prompt_logits[:, :-1],
                    prev_logits,
                ],
                dim=-2,
            ),
            response_length=rlength,
            temperature=self.temperature,
        )

        (
            new_response_input_ids,
            new_response_attention_mask,
        ) = self.prepare_inputs(
            new_response,
            padding_side="right",
        )

        new_response_extra = [
            resp
            + [self.tokenizer.eos_token_id]
            for resp in new_response
        ]

        response_extra = [
            resp
            + [self.tokenizer.eos_token_id]
            for resp in response
        ]

        new_response_outputs = self.model.forward(
            input_ids=new_response_input_ids,
            attention_mask=torch.cat(
                [
                    prompt_attention_mask,
                    new_response_attention_mask,
                ],
                dim=-1,
            ),
            return_dict=True,
            use_cache=True,
            past_key_values=prompt_key_values,
        )

        # logits = new_response_outputs.logits
        # new_response
        (
            prev_lm_likelihood,
            forward_transition_likelihood,
        ) = jacobi_joint_probability(
            prompt_logits=prompt_logits,
            x_logits=prev_logits,
            x=response_extra,
            y=new_response_extra,
            temperature=self.temperature,
        )

        (
            next_lm_likelihood,
            backward_transition_likelihood,
        ) = jacobi_joint_probability(
            prompt_logits=prompt_logits,
            x_logits=new_response_outputs.logits,
            x=new_response_extra,
            y=response_extra,
            temperature=self.temperature,
        )

        mapsum = lambda x: list(map(sum, x))

        chopped_logits = [
            logits_padded[: len(ri)]
            for ri, logits_padded in zip(
                new_response,
                new_response_outputs.logits.cpu().tolist(),
            )
        ]

        return (
            new_response,
            chopped_logits,
            {
                "prev_lm_likelihood": mapsum(
                    prev_lm_likelihood
                ),
                "forward_transition_likelihood": mapsum(
                    forward_transition_likelihood
                ),
                "next_lm_likelihood": mapsum(
                    next_lm_likelihood,
                ),
                "backward_transition_likelihood": mapsum(
                    backward_transition_likelihood,
                ),
            },
        )

    @torch.no_grad()
    def continuation(
        self,
        prompt,
        prefix=None,
        prompt_key_values=None,  # assume o
    ):

        if prefix is None:
            (
                completions,
                transition_scores,
            ) = self.ancestral(
                prompt,
                past_key_values=prompt_key_values,
            )

        else:

            # TODO: It does not nativelly support the functionality I am looking for here.
            # I need to implement it myself.
            # as of now it's not efficient but it has the correct output.

            (
                completions,
                transition_scores,
            ) = self.ancestral(
                prompt=[
                    p + pr
                    for p, pr in zip(
                        prompt, prefix
                    )
                ],
            )

        return (
            completions,
            transition_scores,
        )

    @torch.no_grad()
    def ancestral(
        self,
        prompt,
        past_key_values=None,
    ):

        generated_ids, attention_mask = (
            self.prepare_inputs(prompt)
        )

        batch, T = generated_ids.size()

        eos_token_id = (
            self.tokenizer.eos_token_id
        )

        finished_sequences = torch.zeros(
            generated_ids.shape[0],
            dtype=torch.bool,
            device=self.device_map,
        )

        transition_scores = []
        logits = []
        for i in range(self.max_new_tokens):
            # Forward pass with the model

            # print("s:", generated_ids)
            outputs = self.model(
                input_ids=(
                    generated_ids[:, -1:]
                    if past_key_values
                    is not None
                    else generated_ids
                ),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if past_key_values is None:

                transitions_scores_prompt = (
                    (
                        outputs.logits
                        / self.temperature
                    )
                    .log_softmax(-1)
                    .gather(
                        dim=-1,
                        index=generated_ids.unsqueeze(
                            -1
                        ),
                    )
                    .squeeze(-1)
                )

                transition_scores.extend(
                    transitions_scores_prompt.T
                )

            next_token_logits = (
                outputs.logits[:, -1, :]
                / self.temperature
            )

            logits.append(
                outputs.logits[:, -1, :]
            )

            next_token_probs = softmax(
                next_token_logits,
                dim=-1,
            )

            # sample from the distribution
            next_token_id = (
                torch.multinomial(
                    next_token_probs,
                    num_samples=1,
                )
            )

            # logprobs = next_token_logits.log_softmax(
            #    dim=-1
            # )

            # transition_scores.append(
            #    logprobs.gather(
            #        dim=-1,
            ##        index=next_token_id,
            #    ).squeeze(-1)
            # )

            # Update the attention mask and check for end-of-sequence tokens
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(
                        next_token_id
                    ),
                ],
                dim=1,
            )

            finished_sequences = (
                finished_sequences
                | (
                    next_token_id.squeeze()
                    == eos_token_id
                )
            )

            # Append the next token to the generated sequence
            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token_id,
                ],
                dim=1,
            )

            # Update the past key values
            past_key_values = (
                outputs.past_key_values
            )

            # Break the loop if all sequences are finished
            if finished_sequences.all():
                break

        # prompt_tokens = generated_ids[:, :T]
        generated_ids = (
            generated_ids[:, T:]
            .cpu()
            .tolist()
        )

        # transition_scores = torch.stack(
        #    transition_scores, dim=1
        # )

        # prompt_transition = (
        #    transition_scores[:, :T]
        # )
        # generated_transition = (
        #    transition_scores[:, T:]
        # )
        generated_logits = (
            torch.stack(logits, dim=1)
            .cpu()
            .tolist()
        )

        response = []
        for (sample,) in zip(
            generated_ids,
        ):
            (eos_indices,) = torch.nonzero(
                torch.tensor(sample)
                == eos_token_id,
                as_tuple=True,
            )

            eos_indices = (
                eos_indices.tolist()
            )

            sample = (
                sample[: eos_indices[0] + 1]
                if len(eos_indices) > 0
                else sample
            )

            response.append(sample)

        chopped_logits = [
            logits_padded[: len(ri)]
            for ri, logits_padded in zip(
                response,
                generated_logits,
            )
        ]

        return (
            generated_ids,
            chopped_logits,
            # generated_transition.cpu().tolist(),
        )

    def prepare_inputs(
        self, tokens, padding_side="left"
    ):

        if padding_side == "right":

            self.tokenizer.padding_side = (
                "right"
            )
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        input_dict = self.tokenizer.pad(
            {"input_ids": tokens},
            padding=True,
            return_tensors="pt",
        )

        generated_ids = input_dict[
            "input_ids"
        ].to(self.device_map)

        # Initialize attention mask and finished sequences tracking
        attention_mask = input_dict[
            "attention_mask"
        ].to(self.device_map)

        if padding_side == "right":

            self.tokenizer.padding_side = (
                "left"
            )
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )

        return generated_ids, attention_mask

    def get_starting_cache(
        self, prompt, past_key_values=None
    ):

        generated_ids, attention_mask = (
            self.prepare_inputs(prompt)
        )

        outputs = self.model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return (
            outputs.past_key_values,
            outputs.logits,
            attention_mask,
        )


if __name__ == "__main__":

    import os
    from datasets import load_dataset
    from qflow.utils.data import (
        processhh_data,
    )

    dataset_path = "Anthropic/hh-rlhf"
    temperature = 0.6
    gpu_memory_utilization = 0.8
    model_path = "allenai/tulu-2-7b"  # "openai-community/gpt2"  #  "allenai/tulu-2-7b"
    n = 2

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
        max_new_tokens=800,
    )

    prompt = model.encode(data_iterable)

    """
    prompt_kv = model.get_starting_cache(
        prompt
    )

    prompt_kv2 = model.get_starting_cache(
        prompt,
        past_key_values=prompt_kv,
    )"""

    """response,
    prev_logits,
    prompt_logits,
    prompt_attention_mask,
    prompt_key_values,
    """

    (
        prompt_key_values,
        prompt_logits,
        prompt_attention_mask,
    ) = model.get_starting_cache(prompt)

    generated, generated_logits = (
        model.ancestral(
            prompt=prompt,
        )
    )

    for _ in range(5000):
        print("tokens", generated)
        print(
            "text:",
            model.decode_tokenize(
                generated
            ),
        )

        print("-*-" * 20)
        (
            generated,
            generated_logits,
            scores,
        ) = model.jacobi(
            response=generated,
            prev_logits=generated_logits,
            prompt_logits=prompt_logits,
            prompt_attention_mask=prompt_attention_mask,
            prompt_key_values=prompt_key_values,
        )

        # scores["prev_lm_likelihood"]
        # scores["next_lm_likelihood"]
        # (scores["next_lm_likelihood"] - scores["prev_lm_likelihood"]) + (scores["backward_transition_likelihood"] - scores["forward_transition_likelihood"])
        # scores["forward_transition_likelihood"]
        # scores["backward_transition_likelihood"]

        target_ratio = np.sum(
            scores["next_lm_likelihood"],
            axis=-1,
        ) - np.sum(
            scores["prev_lm_likelihood"],
            axis=-1,
        )

        transition_ratio = np.array(
            scores[
                "backward_transition_likelihood"
            ],
        ) - np.array(
            scores[
                "forward_transition_likelihood"
            ],
        )

        db = np.exp(
            (target_ratio)
            + (transition_ratio)
        )

        acceptance_prob = np.minimum(
            1.0, db
        )

        print(acceptance_prob)

        import pdb

        pdb.set_trace()
    """
     "prev_lm_likelihood": prev_lm_likelihood,
                "forward_transition_likelihood": forward_transition_likelihood,
                "next_lm_likelihood": next_lm_likelihood,
                "backward_transition_likelihood": backward_transition_likelihood,
            },
            
    """
    completions_text = [
        data_iterable[0]["answer"]
    ]  # list of samples.

    completions, transition_scores = (
        model.ancestral(
            prompt  # prompt,
            # completions_text,
        )
    )

    r = model.decode_tokenize(completions)

    prompt_kv = model.get_starting_cache(
        prompt
    )

    prompt_kv2 = model.get_starting_cache(
        prompt,
        past_key_values=prompt_kv,
    )

    # if they are equal it means that the second it kv is the same as the first one.

    assert torch.isclose(
        prompt_kv[0][0], prompt_kv2[0][0]
    ).all()

    import pdb

    pdb.set_trace()

    completion_tokens = model.tokenize(
        completions_text
    )

    model.continuation(
        prompt,
        # prefix=completion_tokens,
        # prompt_key_values=prompt_kv,
    )

    model.continuation(
        prompt,
        prefix=completion_tokens,
        # prompt_key_values=prompt_kv,
    )

    model.continuation(
        prompt,
        prefix=completion_tokens,
        prompt_key_values=prompt_kv,
    )

    completions, transition_scores = (
        model.jacobi(
            prompt,
            response=prompt,
            # completions_text,
        )
    )

    completion_tokens = model.tokenize(
        completions_text
    )

    import pdb

    pdb.set_trace()

# print("passed all tests")
