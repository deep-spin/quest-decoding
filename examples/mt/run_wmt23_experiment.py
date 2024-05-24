import itertools
import json
import logging
import multiprocessing
import os
import uuid
import warnings
from multiprocessing import Pool
from typing import *

import numpy as np
import torch
import transformers
from tqdm import tqdm

from decoding.armcmc import ARMarkovChain, RLHFMarkovChain
from models.tm import ABR_LANGUAGE_MAP
from models.vllm import VLLM
from rewards.mtreward import *
from wmt22_constants import *

transformers.logging.set_verbosity_error()

warnings.filterwarnings("ignore")  # Ignore warnings

logging.getLogger().setLevel(logging.ERROR)  # Show only errors in logging

torch.set_float32_matmul_precision("medium")
logging.basicConfig(level=logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class MCMCGeneration:

    def __init__(
        self,
        language_pair="de-en",
        reward_type="da",
        device_id=2,
        beta=1,
        temperature=1.0,
        steps=1000,
        model_name="alpaca",
        k=32,
        resampling_factor=1,
        reward_batch_size=8,
        gpu_memory_utilization=0.50,
        task="mt",
        **kwargs,
    ):
        self.src_lang, self.tgt_lang = language_pair.split("-")
        self.source_language = ABR_LANGUAGE_MAP[self.src_lang]
        self.target_language = ABR_LANGUAGE_MAP[self.tgt_lang]

        self.reward_type = reward_type
        self.device_id = device_id
        self.k = k
        self.resampling_factor = resampling_factor
        self.reward_batch_size = reward_batch_size

        self.mt = VLLM(
            language_pair=language_pair,
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            device_ids=[device_id],
            task=task,
        )

        self.reward = get_reward_model(
            reward_type=reward_type,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            device_id=device_id,
            batch_size=self.reward_batch_size,
            k=self.k,
        )

        self.beta = beta
        self.temperature = temperature
        self.steps = steps

    def get_name(self):
        return f"{self.src_lang}-{self.tgt_lang}-mcmc-{self.beta}-{self.temperature}-{self.steps}-{self.reward_type}-model-{self.mt.model_path}-k-{self.k}"

    def run(
        self,
        source_sentence,
        reference_sentence,
        candidates=None,
        warm_start=None,
        output_dir=OUTPUT_DIR,
        ext="",
    ):

        if self.reward_type == "mbr":

            if candidates is None:
                candidates = self.mt.ancestral(
                    source_sentence,
                    temperature=self.temperature,
                    n=self.k * self.resampling_factor,
                )

            else:
                assert (
                    len(candidates[0]) == self.k * self.resampling_factor
                ), "Number of candidates must be equal to k"
                assert len(candidates) == len(source_sentence), "Must respect batch"

            self.reward.set_candidates(candidates)

        self.reward.set_reference(reference_sentence)
        self.reward.set_source(source_sentence)

        armc = RLHFMarkovChain(
            source_sentence=source_sentence,
            model=self.mt,
            reward_model=self.reward,
            beta=self.beta,
            temperature=self.temperature,
            # chains=len(source_sentence),
        )

        (
            samples,
            reject_fractions,
            accepted_indices,
            rejected_indices,
            state_path,
        ) = armc.run_chain(
            steps=self.steps,
            # clear_cache_steps=500,
            warm_start=warm_start,
            use_tqdm=True,
        )

        del armc
        results = []
        for i, sample in enumerate(samples):

            extra_info = {"reward_type": self.reward_type}
            if self.reward_type == "mbr":
                extra_info["mbr_samples"] = candidates[i]

            saved_data = {
                "source_sentence": source_sentence[i],
                "reference_sentence": reference_sentence[i],
                "accepted_indices": accepted_indices[i],
                "rejected_indices": rejected_indices[i],
                "src_lang": self.source_language,
                "tgt_lang": self.target_language,
                "sample": sample,
                # "probe": state_path,
                **extra_info,
            }

            # Write to device a json with device id
            # with open(f"{output_dir}/wmt22_test_shard_{i}_node_{ext}.json", "w") as f:
            #    json.dump(saved_data, f)

            results.append(saved_data)

        return results


def mcmc_generation(
    dataset_path: str,
    start_index: int,
    output_dir: str,
    device_ids: List[int] = [0, 2],
    num_samples: int = None,
    batch_size: int = 8,
    **method_kwargs,
):

    # open json in dataset_path
    with open(dataset_path, "r") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[start_index : start_index + num_samples]

    device_id = device_ids[0]

    gen = MCMCGeneration(
        **method_kwargs,
        device_id=device_id,
    )

    src_lang, tgt_lang = method_kwargs["language_pair"].split("-")

    # Break data_split into batches
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    results = []

    # If previous_run_batches is not None, use it for warm start and candidates
    for j, batch in tqdm(list(enumerate(batches))):

        method_input = {}

        reference_sentence = [sample["translation"][tgt_lang] for sample in batch]
        source_sentence = [sample["translation"][src_lang] for sample in batch]

        result = gen.run(
            reference_sentence=reference_sentence,
            source_sentence=source_sentence,
            **method_input,
            output_dir=output_dir,
            ext=f"{j}_{0}",
        )

        results.extend(result)

    # wmt22_test_gen.
    path = f"{output_dir}/wmt22_test_shard_{0}.json"
    # Write to device a json with device id
    with open(path, "w") as f:
        json.dump(results, f)

    return path


def main(
    beta: float = 0.01,
    temperature: float = 0.9,
    steps: int = 50,
    model_name: str = "alpaca",
    k: int = 16,
    language_pair: str = "de-en",
    prefix: str = "",
    num_samples: Union[None, int] = None,
    start_index: int = 0,
    device_ids: List[int] = [0, 1, 2, 3, 4, 5],
    batch_size: int = 8,
    previous_run_dir: Union[None, str] = None,
    reward_type: str = "da",
    reward_model_checkpoint: str = "Unbabel/wmt22-comet-da",
    dataset_path_dir: str = DATASET_PATH_DIR,
    seed: int = 0,
    output_dir: str = OUTPUT_DIR,
    gpu_memory_utilization=0.6,
    reward_batch_size=32,
    task="mt",
):

    method = "mcmc"

    torch.manual_seed(seed)
    np.random.seed(seed)
    multiprocessing.set_start_method("spawn", force=True)

    lp_order = "".join(
        sorted(language_pair.split("-"), key=lambda x: 1 if "en" == x else 0)
    )

    method_kwargs = {
        "language_pair": language_pair,
        "reward_type": reward_type,
        "beta": beta,
        "temperature": temperature,
        "steps": steps,
        "model_name": model_name,
        "k": k,
        "reward_model_checkpoint": reward_model_checkpoint,
        "dataset_path": f"{dataset_path_dir}/{lp_order}/test.{language_pair}.json",
        "method": method,
        "previous_run_dir": previous_run_dir,
        "task": task,
    }

    unique_id = prefix + str(uuid.uuid4())
    output_dir = f"{output_dir}/{unique_id}"

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump(method_kwargs, f)

    mcmc_generation(
        device_ids=device_ids,
        num_samples=num_samples,
        start_index=start_index,
        output_dir=output_dir,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        reward_batch_size=reward_batch_size,
        **method_kwargs,
    )


if __name__ == "__main__":

    import fire

    fire.Fire(main)
