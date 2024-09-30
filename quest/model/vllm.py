from quest.model.base import (
    LocalLanguageModel,
    DEFAULT_TEMPLATE,
)
from langchain.prompts import PromptTemplate
import numpy as np
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from quest.utils.list import (
    flatten_list,
    unflatten_list,
)


class VLLM(LocalLanguageModel):
    def __init__(
        self,
        model_path: str,
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=["</s>"],  # ["\n"],
        temperature=1.0,
        skip_special_tokens=False,
        **llm_kwargs
    ):
        super().__init__(
            model_path=model_path,
            prompt_template=prompt_template,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            stop_tokens=stop_tokens,
            skip_special_tokens=skip_special_tokens,
        )

        self.model = LLM(
            model=model_path, **llm_kwargs
        )

    def continuation(
        self, prompt, prefix=None
    ):

        if prefix is None:
            input_data = prompt
        else:
            input_data = [
                x[0] + x[1]
                for x in zip(prompt, prefix)
            ]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            logprobs=1,
            max_tokens=self.max_new_tokens,
            stop=self.stop_tokens,
            include_stop_str_in_output=True,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=False,
        )

        outputs = self.model.generate(
            prompt_token_ids=input_data,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        completion = [
            list(out.outputs[0].token_ids)
            for out in outputs
        ]

        scores = [
            [
                (lxi[xi].logprob)
                for xi, lxi in zip(
                    compl,
                    out.outputs[0].logprobs,
                )
            ]
            for compl, out in zip(
                completion, outputs
            )
        ]

        return completion, scores

    def evaluate_continuation(
        self, prompt, completion_text
    ):

        input_ids = []
        completion_ids = []
        ns = []
        for prompt_ids, completion in zip(
            prompt, completion_text
        ):

            ns.append(len(prompt_ids))
            completion_ids_i = (
                self.tokenize([completion])[
                    0
                ][1:]
            )

            input_ids_i = (
                prompt_ids
                + completion_ids_i
                + [
                    self.tokenizer.eos_token_id
                ]
            )

            completion_ids.append(
                completion_ids_i
                + [
                    self.tokenizer.eos_token_id
                ]
            )
            input_ids.append(input_ids_i)

        sampling_params = SamplingParams(
            # logprobs=1,
            prompt_logprobs=0,
            max_tokens=1,
            temperature=self.temperature,
        )

        outputs = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        scores = [
            [
                (lxi[xi].logprob)
                for xi, lxi in zip(
                    compl[1:],
                    out.prompt_logprobs[1:],
                )
            ][n - 1 :]
            for compl, out, n in zip(
                input_ids, outputs, ns
            )
        ]

        return completion_ids, scores

    def ancestral(
        self,
        input_data,
        # temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        use_beam_search=False,
        best_of=None,
        n=1,
    ):

        prompt_txt = flatten_list(
            [
                [self.get_prompt(**data)]
                * n
                for data in input_data
            ]
        )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            n=1,
            top_p=top_p,
            min_p=min_p,
            best_of=best_of,
            use_beam_search=use_beam_search,
            max_tokens=self.max_new_tokens,
            stop=self.stop_tokens,
            include_stop_str_in_output=True,
            skip_special_tokens=self.skip_special_tokens,
        )

        responses = self.model.generate(
            prompt_txt, sampling_params
        )

        completions = []
        for out in responses:
            out_inst = []
            for i in range(
                len(out.outputs)
            ):
                out_inst.append(
                    out.outputs[
                        i
                    ].text  # .rstrip("\n")
                )
            completions.extend(out_inst)

        completions = unflatten_list(
            completions,
            [n] * len(input_data),
        )

        return completions
