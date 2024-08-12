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
    beta: float = 0.1,
    temperature: float = 0.8,
    steps: int = 128,
    llm: str = "alma",
    language_pair: str = "en-cs",
    reward_model_checkpoint: str = "Unbabel/wmt23-cometkiwi-da-xl",
    seed: int = 0,
    gpu_memory_utilization=0.6,
    reward_batch_size=8,
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
            "beta": beta,
            "steps": steps,
            "temperature": temperature,
            "model_path": llms[llm]["path"],
            "reward_model_path": reward_model_checkpoint,
            "variant": "quest",
            "reward_type": "contextual",
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
        skip_special_tokens=True,
    )

    reward = QEModel(
        model_path=reward_model_checkpoint,
        batch_size=reward_batch_size,
        device_count=device_count,
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

    sources = [
        sample["source_sentence"]
        for sample in input_data
    ]
    reward.set_sources(sources)

    chain_outputs = Quest(
        input_data=input_data,
        proposal=SuffixProposal(
            model=model
        ),
        reward=reward,
        beta=beta,
    ).run(steps=steps, use_tqdm=True)

    outputs = []
    for i in range(len(input_data)):
        outputs.append(
            [
                {
                    "t": s["t"],
                    **{
                        k: v[i]
                        for k, v in s.items()
                        if k != "t"
                    },
                }
                for s in chain_outputs.state_path
            ]
        )

    experiment.add_instances(
        inputs=input_data,
        outputs=outputs,
    )

    beta = experiment.meta["beta"]
    eval_key = reward.get_name()
    instances = experiment.instances
    scores = [
        {
            "scores": [
                o["reward"] * beta
                for o in i.outputs
            ]
        }
        for i in instances
    ]

    experiment.add_eval(
        eval_key,
        scores,
    )

    experiment.save(save_path)


if __name__ == "__main__":

    import fire

    fire.Fire(main)
