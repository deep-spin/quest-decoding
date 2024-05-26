import json
import logging
import os
import warnings
from typing import *
from langchain.prompts import PromptTemplate
import numpy as np
import transformers
from datasets import load_dataset

from quest.decoding import Quest

# from models.tm import ABR_LANGUAGE_MAP
from quest.model.vllm import VLLM
from quest.reward.qe import QEModel

# from wmt22_constants import *

ABR_LANGUAGE_MAP = {
    "pt": "Portuguese",
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese",
}


transformers.logging.set_verbosity_error()

warnings.filterwarnings("ignore")  # Ignore warnings

logging.getLogger().setLevel(logging.ERROR)  # Show only errors in logging

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


def generate(
    batch_size: int = 512,
    gpu_memory_utilization=0.6,
    llm: str = "alma",
    beta: float = 0.1,
    temperature: float = 0.8,
    reward_model_checkpoint="Unbabel/wmt23-cometkiwi-da-xl",
    steps: int = 50,
    reward_batch_size: int = 8,
    device_count: int = 1,
    language_pair="en-de",
):

    src_lang, tgt_lang = language_pair.split("-")
    data = load_dataset("haoranxu/WMT23-Test", language_pair, split="test")
    input_data = [
        {
            "reference_sentence": sample[language_pair][tgt_lang],
            "source_sentence": sample[language_pair][src_lang],
            "source_language": ABR_LANGUAGE_MAP[src_lang],
            "target_language": ABR_LANGUAGE_MAP[tgt_lang],
        }
        for sample in data
    ]

    model = VLLM(
        model_path=llms[llm]["path"],
        prompt_template=llms[llm]["prompt"],
        download_dir=os.environ["HF_HOME"],
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=device_count,
    )

    reward = QEModel(
        reward_model_checkpoint, batch_size=reward_batch_size, device_count=device_count
    )  # sentiment model.

    reward.set_sources([sample["source_sentence"] for sample in input_data])

    output = Quest(
        input_data=input_data,
        model=model,
        reward=reward,
        beta=beta,
        batch_size=batch_size,
    ).run(steps=steps, use_tqdm=True)

    return output.samples


def main(
    beta: float = 0.01,
    temperature: float = 0.9,
    steps: int = 50,
    llm: str = "alma",
    language_pair: str = "de-en",
    batch_size: int = 8,
    reward_model_checkpoint: str = "Unbabel/wmt23-cometkiwi-da-xl",
    seed: int = 0,
    gpu_memory_utilization=0.6,
    reward_batch_size=8,
    device_count=1,
):
    np.random.seed(seed)

    method_kwargs = {
        "language_pair": language_pair,
        "beta": beta,
        "temperature": temperature,
        "steps": steps,
        "reward_model_checkpoint": reward_model_checkpoint,
        "llm": llm,
    }

    samples = generate(
        device_count=device_count,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        reward_batch_size=reward_batch_size,
        **method_kwargs,
    )

    path = f"quest_outputs_{steps}_{llm}_{language_pair}_{temperature}_{beta}.json"
    # Write to device a json with device id
    with open(path, "w") as f:
        json.dump(samples, f)

    return path


if __name__ == "__main__":

    import fire

    fire.Fire(main)
