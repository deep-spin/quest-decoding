import json
import logging
import os
import warnings
from typing import *
from langchain.prompts import PromptTemplate
import numpy as np
import transformers
from datasets import load_dataset
from quest import (
    Quest,
    SuffixProposal,
    RLHFSuffixProposal,
)
from datetime import datetime

from quest.model.vllm import VLLM
from quest.reward.mt import QEModel
from quest.reward.base import (
    ConstantReward,
)

from expkit import Exp, DiskStorage
import torch

from quest.model.remote import RemoteVLLM

from quest.reward.remote import RemoteReward
from qflow.serving.registry import ModelRegistry
from literegistry import FileSystemKVStore, RegistryClient

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
    "hi": "Hindi",
    "ja": "Japanese",
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
    "euro": {
        "path": "utter-project/EuroLLM-1.7B-Instruct",
        "prompt": PromptTemplate.from_template(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTranslate the following {source_language} source text to {target_language}:\n{source_language}: {source_sentence}\n{target_language}: <|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    "llama3-3b": {
        "path": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": PromptTemplate.from_template(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 12 Feb 2025\n\nTranslate the following {source_language} source text to {target_language}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{source_language}: {source_sentence}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_language}: "
        ),
    },
}

# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 12 Feb 2025\n\nTranslate the following {source_language} source text to {target_language}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{source_language}: {source_sentence}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_language}: '


def load_wmt_data(language_pair: str = "en-de", year="23"):
    src_lang, tgt_lang = language_pair.split("-")
    data = load_dataset(
        f"graf/WMT{year}-Test",
        split="test",
    ).filter(lambda x: x["lp"] == language_pair)

    assert len(data) > 0, f"No data found for {language_pair} in WMT{year}-Test"
    print("Num of examples:", len(data))
    input_data = [
        {
            "reference_sentence": sample["tgt"],
            "source_sentence": sample["src"],
            # "lp": sample["lp"],
            "source_language": ABR_LANGUAGE_MAP[src_lang],
            "target_language": ABR_LANGUAGE_MAP[tgt_lang],
        }
        for sample in data
    ]

    return input_data


def process_batch_outputs(
    chain_outputs: Any, batch_size: int
) -> List[List[Dict[str, Any]]]:
    """
    Processes batch outputs from a Quest chain into a standardized format.
    """
    outputs = []
    for i in range(batch_size):
        outputs.append(
            [
                {
                    "t": s["t"],
                    **{k: v[i] for k, v in s.items() if k != "t"},
                }
                for s in chain_outputs.state_path
            ]
        )
    return outputs


def main(
    beta: float = 1.0,
    temperature: float = 1.0,
    steps: int = 128,
    llm: str = "euro",
    language_pair: str = "en-cs",
    reward_model_checkpoint: str = "Unbabel/XCOMET-XL",  # "Unbabel/wmt23-cometkiwi-da-xl",#Unbabel/XCOMET-XL
    seed: int = 0,
    gpu_memory_utilization=0.6,
    reward_batch_size=1024,  # 1024,
    device_count=2,
    reward_device_count=1,
    stop_tokens=[],
    max_new_tokens=800,
    max_prompt_length=1200,
    reward_device=0,
    save_path: str = "mt-outputs/",
    year: str = "24",
    n: int = 2000,
):
    np.random.seed(seed)

    input_data = load_wmt_data(language_pair, year=year)

    experiment = Exp(
        storage=DiskStorage(save_path, "rw"),
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
            "split": "test",
            "n": min(len(input_data), n),
            "skip_special_tokens": True,
            "dataset": f"graf/WMT{year}-Test",
            "corrected": 1,
        },
    )

    model_args = {
        "model_path": llms[llm]["path"],
        "download_dir": os.environ.get("HF_HOME", "/tmp/"),
        "stop_tokens": stop_tokens,
        "temperature": temperature,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": "bfloat16",
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": max_prompt_length,
        "tensor_parallel_size": device_count,
        "prompt_template": llms[llm]["prompt"],
        "skip_special_tokens": True,
    }

    registry = RegistryClient(
        store=FileSystemKVStore("/gscratch/ark/graf/registry"),
        max_history=3600,
        cache_ttl=60 * 5,
        service_type="model_path",
    )

    # registry.models()
    # models = async_wrapper(registry.models())
    # targets = async_wrapper(registry.get_all(llms[llm]["path"]))
    # print(targets)

    model = RemoteVLLM(registry=registry, **model_args)

    input_data = [
        {
            "prompt": model.get_prompt(**x),
            **x,
        }
        for x in input_data
    ][:n]

    reward = RemoteReward(
        registry=registry,
        model_path=experiment.meta["reward_model_path"],
        reward_type="qe",
        batch_size=reward_batch_size,
    )

    # import pdb

    # pdb.set_trace()

    """
    reward = QEModel(
        model_path=reward_model_checkpoint,
        batch_size=reward_batch_size,
        device_count=reward_device_count,
        devices=(np.arange(reward_device_count) + reward_device).tolist(),
    )
    """

    sources = [sample["source_sentence"] for sample in input_data]

    reward.set_context(sources)

    chain = Quest(
        input_data=input_data,
        proposal=RLHFSuffixProposal(model=model, reward=reward),
        # reward=reward,
        beta=beta,
    )

    chain_outputs = chain.run(steps=steps, use_tqdm=True)

    outputs = process_batch_outputs(chain_outputs, len(input_data))

    experiment.add_instances(
        inputs=input_data,
        outputs=outputs,
    )


if __name__ == "__main__":

    import fire

    fire.Fire(main)
