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

from quest.model.vllm import VLLM
from quest.reward.mt import QEModel

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


def load_wmt23_data(
    language_pair: str = "en-de",
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

    return input_data


def generate(
    llm: str = "alma",
    beta: float = 0.1,
    temperature: float = 0.8,
    reward_model_checkpoint="Unbabel/wmt23-cometkiwi-da-xl",
    steps: int = 50,
    language_pair="en-de",
    reward_batch_size: int = 8,
    device_count: int = 1,
    gpu_memory_utilization=0.6,
):

    input_data = load_wmt23_data(language_pair)

    model = VLLM(
        model_path=llms[llm]["path"],
        prompt_template=llms[llm]["prompt"],
        download_dir=os.environ["HF_HOME"],
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=device_count,
        temperature=temperature,
    )

    reward = QEModel(
        model_path=reward_model_checkpoint,
        batch_size=reward_batch_size,
        device_count=device_count,
    )

    sources = [sample["source_sentence"] for sample in input_data]
    reward.set_sources(sources)

    output = Quest(
        input_data=input_data,
        model=model,
        reward=reward,
        beta=beta,
    ).run(steps=steps, use_tqdm=True)

    return [
        {"outputs": outputs, "source": src}
        for outputs, src in zip(output.samples, sources)
    ]


def main(
    beta: float = 0.01,
    temperature: float = 0.9,
    steps: int = 50,
    llm: str = "alma",
    language_pair: str = "de-en",
    reward_model_checkpoint: str = "Unbabel/wmt23-cometkiwi-da-xl",
    seed: int = 0,
    gpu_memory_utilization=0.6,
    reward_batch_size=8,
    device_count=1,
    output_file_name="outputs.json",
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
        gpu_memory_utilization=gpu_memory_utilization,
        reward_batch_size=reward_batch_size,
        **method_kwargs,
    )

    # Write to device a json with device id
    with open(output_file_name, "w") as f:
        json.dump(samples, f)


if __name__ == "__main__":

    import fire

    fire.Fire(main)
