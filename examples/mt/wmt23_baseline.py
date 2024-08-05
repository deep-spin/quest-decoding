import json
import logging
import os
import warnings
from typing import *
from langchain.prompts import PromptTemplate
import numpy as np
import transformers
from datasets import load_dataset
from quest import Quest, SuffixProposal
from datetime import datetime

from quest.model.vllm import VLLM
from quest.reward.mt import QEModel

from expkit import Exp
import torch

torch.set_float32_matmul_precision("medium")

ABR_LANGUAGE_MAP = {
    "pt": "Portuguese",
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese",
    "cs": "Czech",
    "is": "Icelandic",
}

transformers.logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore"
)  # Ignore warnings

logging.getLogger().setLevel(
    logging.ERROR
)  # Show only errors in logging
logging.basicConfig(level=logging.ERROR)

llms = {
    "alma": {
        "path": "haoranxu/ALMA-7B",
        "prompt": PromptTemplate.from_template(
            "Translate this from {source_language} to {target_language}:\n{source_language}: {source_sentence}\n{target_language}:"
        ),
    },
    "tower": {
        "path": "Unbabel/TowerInstruct-7B-v0.2",
        "prompt": PromptTemplate.from_template(
            "<|im_start|>user\nTranslate the following {source_language} source text to {target_language}:\n{source_language}: {source_sentence}\n{target_language}: <|im_end|>\n<|im_start|>assistant\n"
        ),
    },
}


def load_wmt_data(
    language_pair: str = "en-de", year="23"
):
    src_lang, tgt_lang = (
        language_pair.split("-")
    )
    data = load_dataset(
        f"haoranxu/WMT{year}-Test",
        language_pair,
        split="test",
    )
    input_data = [
        {
            "reference_sentence": sample[
                language_pair
            ][tgt_lang],
            "source_sentence": sample[
                language_pair
            ][src_lang],
            "source_language": ABR_LANGUAGE_MAP[
                src_lang
            ],
            "target_language": ABR_LANGUAGE_MAP[
                tgt_lang
            ],
        }
        for sample in data
    ]

    return input_data


def main(
    temperature: float = 0.9,
    steps: int = 128,
    llm: str = "alma",
    language_pair: str = "de-en",
    seed: int = 0,
    gpu_memory_utilization=0.85,
    device_count=1,
    stop_tokens=[],
    max_new_tokens=800,
    max_prompt_length=1200,
    save_path: str = "mt-outputs/",
    year: str = "23",
):
    np.random.seed(seed)

    experiment = Exp(
        meta={
            "steps": steps,
            "temperature": temperature,
            "model_path": llms[llm]["path"],
            "variant": "ancestral",
            "stop_tokens": stop_tokens,
            "index": "uniform",
            "max_new_tokens": max_new_tokens,
            "max_prompt_length": max_prompt_length,
            "at": datetime.now().isoformat(),
            "language_pair": language_pair,
            "llm": llm,
        }
    )

    model = VLLM(
        model_path=llms[llm]["path"],
        download_dir=os.environ.get(
            "HF_HOME", "/tmp/"
        ),
        stop_tokens=stop_tokens,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        max_new_tokens=max_new_tokens,  # 100
        max_prompt_length=max_prompt_length,  # 600
        tensor_parallel_size=device_count,
        prompt_template=llms[llm]["prompt"],
    )

    input_data = load_wmt_data(
        language_pair, year=year
    )

    input_data = [
        {
            "prompt": model.get_prompt(**x),
            **x,
        }
        for x in input_data
    ]

    completions_txt = model.ancestral(
        input_data, n=steps
    )

    outputs = []
    for instance_txt in completions_txt:
        outputs.append(
            [
                {"text": state_t}
                for state_t in instance_txt
            ]
        )

    experiment.add_instances(
        inputs=input_data,
        outputs=outputs,
    )

    experiment.save(save_path)


if __name__ == "__main__":

    import fire

    fire.Fire(main)
