from quest.model.base import LanguageModel
from langchain.prompts import PromptTemplate
import numpy as np
from langchain.prompts import PromptTemplate
from transformers import  AutoTokenizer

from vllm import LLM, SamplingParams



class VLLM(LanguageModel):
    def __init__(
        self,
        model_path:str,
        prompt_template: PromptTemplate,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        **llm_kwargs
    ):
        super().__init__(
           prompt_template 
        )

        self.model = LLM(
            model=model_path,
            **llm_kwargs
        )
        self.max_new_tokens=max_new_tokens
        self.max_prompt_length=max_prompt_length
        self.stop_tokens = stop_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )  # THIS IS ACTUALLY REALLY IMPORTANT :) THIS HIDDEN NIGHTMARE DONT USE EOS. - w/ AR models in batch we may have padding in the beginig - obvious reason left to right gen.
            self.tokenizer.pad_token = self.tokenizer.bos_token

    def encode(self, input_data):

        tokenized_data = self.tokenize(
                [self.get_prompt(**data) for data in input_data],
                #max_length=self.max_prompt_length,
                #truncation=True,
            )#.input_ids

        return tokenized_data

    def tokenize(self, prompt):
        
        return [ self.tokenizer.encode(
            p,
            max_length=self.max_prompt_length,
            truncation=True,
            #return_tensors="np",q
        ) for p in prompt ]
        
    def decode_tokenize(self, ids):
        return self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=True,
        )
        
    def continuation(self, prompt, prefix=None, temperature=1.0):

        if prefix is None:
            input_data = prompt
        else:
            input_data = [x[0] + x[1] for x in zip(prompt, prefix)]

        sampling_params = SamplingParams(
            temperature=temperature,
            logprobs=0,
            max_tokens=self.max_new_tokens,
            stop=self.stop_tokens,
            include_stop_str_in_output=False,
        )

        outputs = self.model.generate(
            prompt_token_ids=input_data,
            sampling_params=sampling_params,
            use_tqdm=False,
        )


        completion = [out.outputs[0].token_ids for out in outputs]
        scores = [
            [lxi[xi] for xi, lxi in zip(compl, out.outputs[0].logprobs)]
            for compl, out in zip(completion, outputs)
        ]

        return completion, scores

    def evaluate_continuation(self, prompt, completion_text, temperature=1.0):

        input_ids = []
        completion_ids = []
        ns = []
        for prompt_ids, completion in zip(prompt, completion_text):

            ns.append(len(prompt_ids))
            completion_ids_i = self.tokenize(completion)[0, 1:]

            input_ids_i = np.concatenate(
                [
                    np.array(prompt_ids),
                    completion_ids_i,  # [:, 1:],
                    np.array(
                        [self.tokenizer.eos_token_id],
                    ),
                ],
                dim=0,
            ).tolist()
            completion_ids.append(
                completion_ids_i.tolist() + [self.tokenizer.eos_token_id]
            )
            input_ids.append(input_ids_i)

        sampling_params = SamplingParams(
            logprobs=0,
            prompt_logprobs=0,
            max_tokens=1,
            temperature=temperature,
        )

        outputs = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        scores = [
            [lxi[xi] for xi, lxi in zip(compl[1:], out.prompt_logprobs[1:])][n - 1 :]
            for compl, out, n in zip(input_ids, outputs, ns)
        ]

        return completion_ids, scores
