import torch
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

from typing import List

DEFAULT_TEMPLATE = PromptTemplate.from_template("{prompt}")


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


class HF2(LocalLanguageModel):

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
        use_flash_attention_2=True,
        dtype=torch.bfloat16,
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
        self.device_map = f"cuda:{device_ids[0]}"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            pad_token_id=self.tokenizer.pad_token_id,
            cache_dir=download_dir,
            use_flash_attention_2=use_flash_attention_2,
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
        # torch.backends.cuda.matmul.allow_tf32 = (
        #    True  # allow tf32 on matmul
        # )
        # torch.backends.cudnn.allow_tf32 = (
        #    True  # allow tf32 on cudnn
        # )

        # self.model = torch.compile(
        #    self.model
        # )

        self.stop_words_ids = (
            self.tokenizer.additional_special_tokens_ids
            + [self.tokenizer.eos_token_id]
            + [
                self.tokenizer.encode(
                    stop_word,
                    add_prefix_space=False,
                    add_special_tokens=False,
                )  # by default the tokenizer is sending spaces. We don't want that
                for stop_word in (stop_tokens)  # , "[/INST]", "[INST]"
            ]
        )

    @torch.no_grad()
    def test_cache_system(
        self,
    ):

        return None

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
                prompt=[p + pr for p, pr in zip(prompt, prefix)],
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

        generated_ids, attention_mask = self.prepare_inputs(prompt)

        batch, T = generated_ids.size()

        eos_token_id = self.tokenizer.eos_token_id

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
                    if past_key_values is not None
                    else generated_ids
                ),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if past_key_values is None:

                transitions_scores_prompt = (
                    (outputs.logits / self.temperature)
                    .log_softmax(-1)
                    .gather(
                        dim=-1,
                        index=generated_ids.unsqueeze(-1),
                    )
                    .squeeze(-1)
                )

                transition_scores.extend(transitions_scores_prompt.T)

            next_token_logits = outputs.logits[:, -1, :] / self.temperature

            logits.append(outputs.logits[:, -1, :])

            next_token_probs = softmax(
                next_token_logits,
                dim=-1,
            )

            # sample from the distribution
            next_token_id = torch.multinomial(
                next_token_probs,
                num_samples=1,
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
                    torch.ones_like(next_token_id),
                ],
                dim=1,
            )

            for stop_word_id in self.stop_words_ids:
                finished_sequences = finished_sequences | (
                    next_token_id.squeeze() == stop_word_id
                )

            # Append the next token to the generated sequence
            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token_id,
                ],
                dim=1,
            )

            """for ids in generated_ids:
                print(
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                    ).split(
                        "<|assistant|>"
                    )[-1]
                )
                print("--" * 20)
            """

            # Update the past key values
            past_key_values = outputs.past_key_values

            del outputs
            torch.cuda.empty_cache()

            # Break the loop if all sequences are finished
            if finished_sequences.all():
                break

        # prompt_tokens = generated_ids[:, :T]
        generated_ids = generated_ids[:, T:].cpu().tolist()

        # transition_scores = torch.stack(
        #    transition_scores, dim=1
        # )

        # prompt_transition = (
        #    transition_scores[:, :T]
        # )
        # generated_transition = (
        #    transition_scores[:, T:]
        # )
        generated_logits = torch.stack(logits, dim=1).cpu().tolist()

        response = []
        for (sample,) in zip(
            generated_ids,
        ):
            (eos_indices,) = torch.nonzero(
                torch.tensor(sample) == eos_token_id,
                as_tuple=True,
            )

            eos_indices = eos_indices.tolist()

            sample = sample[: eos_indices[0] + 1] if len(eos_indices) > 0 else sample

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

    def prepare_inputs(self, tokens, padding_side="left"):

        if padding_side == "right":

            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        input_dict = self.tokenizer.pad(
            {"input_ids": tokens},
            padding=True,
            return_tensors="pt",
        )

        generated_ids = input_dict["input_ids"].to(self.device_map)

        # Initialize attention mask and finished sequences tracking
        attention_mask = input_dict["attention_mask"].to(self.device_map)

        if padding_side == "right":

            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id

        return generated_ids, attention_mask

    def get_starting_cache(self, prompt, past_key_values=None):

        generated_ids, attention_mask = self.prepare_inputs(prompt)

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

    def evaluate_continuation(
        self,
        prompt: List[List[int]],
        completion_text: List[str],
    ):
        (
            generated_ids,
            attention_mask,
        ) = self.prepare_inputs(prompt)

        completion_tokens = self.tokenize(completion_text)

        import pdb

        pdb.set_trace()

        completion_ids = self.tokenizer.encode(
            completion_tokens,
            add_special_tokens=False,
        )

        input_ids = generated_ids.clone()
        input_ids[:, -len(completion_ids) :] = torch.tensor(
            completion_ids, dtype=torch.long
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        logits = outputs.logits[:, -len(completion_ids) :, :]
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        return logprobs


if __name__ == "__main__":

    import os
    from datasets import load_dataset
    from qflow.utils.data import (
        processhh_data,
    )

    dataset_path = "Anthropic/hh-rlhf"
    temperature = 0.001
    # gpu_memory_utilization = 0.8
    model_path = "google/gemma-2-2b-it"  # "allenai/tulu-2-7b"  # "openai-community/gpt2"  #  "allenai/tulu-2-7b"
    n = 1

    ds = load_dataset(dataset_path, split="test")

    psd = ds.map(processhh_data)

    data_iterable = list(psd)[:n]

    model = HF(
        model_path=model_path,
        download_dir=os.environ.get("HF_HOME", "/tmp/"),
        temperature=temperature,
        dtype=torch.bfloat16,
        max_new_tokens=200,
        max_prompt_length=800,
    )

    prompt = model.encode(data_iterable)

    import pdb

    pdb.set_trace()
    prompt_kv = model.get_starting_cache(prompt)

    prefix = [
        model.tokenizer.encode(
            " ",
            add_special_tokens=False,
            padding=False,
        )
    ] * n

    # model.sample(
    #    input_ids,
    #    past=prompt_kv,
    # )
    """
    outputs.past_key_values,
    outputs.logits,
    attention_mask,
    """
    import pdb

    pdb.set_trace()

    completions, transition_scores_ = model.continuation(
        prompt,
        prefix=prefix,
        prompt_key_values=None,  # assume o
    )
    """model.continuation(
        prompt,
        prefix=,
        prompt_key_values=None,  # assume o
    )"""

    import pdb

    pdb.set_trace()

    generated, generated_logits = model.ancestral(
        prompt=prefix,
        past_key_values=prompt_kv,
    )

    import pdb

    pdb.set_trace()
