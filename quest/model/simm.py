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

from quest.reward.base import Reward

from quest.proposal import JointRLHFSuffixProposal, RLHFSuffixProposal
from quest.core import Quest
import numpy as np

from punica.misc.base import c

# from punica.misc.text import MultiLoraInferenceText
from typing import List

DEFAULT_TEMPLATE = PromptTemplate.from_template("{prompt}")

ACTIVE_SERVERS = {}


class Punica(LocalLanguageModel):

    def __init__(
        self,
        lora_weights=None,
        model_path: str = "meta-llama/Llama-2-7b-hf",
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        temperature=1.0,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        # use_flash_attention_2=True,
        dtype=torch.float16,
        request_type="joint",
        lora_name="proposal",
        batch_size=16,
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
        # self.device_map = f"cuda:{device_ids[0]}"
        self.device = torch.device(f"cuda:{device_ids[0]}")
        self.lora_name = lora_name

        assert lora_weights is not None, "lora_weights must be provided."

        load_value_head = "reward" in lora_weights

        # with torch.cuda.device(self.device):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        did = device_ids[0]

        ideal_match_server = request_type + str(did)
        compatible_match_server = "joint" + str(did)

        if (ideal_match_server not in ACTIVE_SERVERS) and (
            compatible_match_server not in ACTIVE_SERVERS
        ):
            self.engine = MultiLoraInference(
                lora_weights,
                base_model=model_path,
                dtype=dtype,
                device=self.device,
                download_dir=download_dir,
                maxlen=max_new_tokens + max_prompt_length,
                load_value_head=load_value_head,
                batch_size=batch_size,
            )

            ACTIVE_SERVERS[ideal_match_server] = self.engine
        elif compatible_match_server in ACTIVE_SERVERS:
            self.engine = ACTIVE_SERVERS[compatible_match_server]
        else:
            self.engine = ACTIVE_SERVERS[ideal_match_server]

        self.request_type = request_type

    def continuation(
        self,
        prompt,
        prefix=None,
        prompt_key_values=None,  # assume o
    ):

        if prefix is None:
            return self.ancestral(
                prompt,
                past_key_values=prompt_key_values,
            )

        else:

            # TODO: It does not nativelly support the functionality I am looking for here.
            # I need to implement it myself.
            # as of now it's not efficient but it has the correct output.

            return self.ancestral(
                prompt=[p + pr for p, pr in zip(prompt, prefix)],
            )

    def ancestral(
        self,
        prompt,
        past_key_values=None,
    ):

        with torch.cuda.device(self.device):
            # torch.cuda.synchronize(self.device)

            output = (self.engine.generate)(
                prompts=prompt,
                temperature=self.temperature,
                lora_name=self.lora_name,
                request_type=self.request_type,
            )

            # torch.cuda.synchronize(self.device)

        return output


## Single input multiple models (SIMM)
class PunicaSIMM(Punica):
    def __init__(
        self,
        proposal_path: str,
        reward_path: str,
        model_path: str = "meta-llama/Llama-2-7b-hf",
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        temperature=1.0,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        dtype=torch.float16,
        batch_size=16,
        **kwargs,
    ):

        super().__init__(
            # proposal_path=proposal_path,
            model_path=model_path,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            stop_tokens=stop_tokens,
            temperature=temperature,
            device_ids=device_ids,
            download_dir=download_dir,
            seed=seed,
            dtype=dtype,
            lora_weights={
                "reward": reward_path,
                "proposal": proposal_path,
            },
            request_type="joint",
            batch_size=batch_size,
        )


class PunicaSISM(Punica):

    def __init__(
        self,
        proposal_path: str,
        model_path: str = "meta-llama/Llama-2-7b-hf",
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        stop_tokens=[],  # ["\n"],
        temperature=1.0,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        # use_flash_attention_2=True,
        dtype=torch.float16,
        batch_size=16,
        **kwargs,
    ):

        super().__init__(
            lora_weights={
                "proposal": proposal_path,
            },
            model_path=model_path,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            stop_tokens=stop_tokens,
            temperature=temperature,
            device_ids=device_ids,
            download_dir=download_dir,
            seed=seed,
            dtype=dtype,
            request_type="text",
            batch_size=batch_size,
        )


class PunicaR(Punica):

    def __init__(
        self,
        reward_path: str,
        model_path: str = "meta-llama/Llama-2-7b-hf",
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        dtype=torch.float16,
        batch_size=16,
        **kwargs,
    ):

        super().__init__(
            lora_weights={
                "reward": reward_path,
            },
            model_path=model_path,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            device_ids=device_ids,
            download_dir=download_dir,
            seed=seed,
            dtype=dtype,
            request_type="reward",
            lora_name="reward",
            batch_size=batch_size,
        )


class ContextualPunicaModel(Reward):

    # applies the model only on outputs r(y)
    """
    RewardModel class represents a reward model based on a pre-trained Hugging Face model.

    Args:
        model_path (str): The path to the pre-trained model.
        batch_size (int, optional): The batch size for inference. Defaults to 32.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.

    Attributes:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to use for inference.

    Methods:
        evaluate(candidates: List[str]) -> List[float]:
            Evaluates a list of candidate sequences and returns a list of reward values.

    """

    def __init__(
        self,
        reward_path: str,
        model_path: str = "meta-llama/Llama-2-7b-hf",
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens=600,
        max_prompt_length=300,
        dtype=torch.float16,
        device_ids=[0],
        download_dir="/tmp/",
        seed=0,
        clamp: float = 40,
        batch_size=16,
        **kwargs,
    ):

        super().__init__(f"rm:{model_path}")

        self.engine = PunicaR(
            reward_path=reward_path,
            model_path=model_path,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            device_ids=device_ids,
            download_dir=download_dir,
            seed=seed,
            dtype=dtype,
            batch_size=batch_size,
            **kwargs,
        )

        self.clamp = clamp
        self.prompt_txt = None

    def set_context(self, context: List[str]):
        self.prompt_txt = context

    def evaluate(
        self,
        candidates: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int]): The list of indices of accepted candidates.
            batch_size (int, optional): The batch size for inference. Defaults to 32.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        context_candidates = [
            self.engine.tokenizer.encode(prompt + str2)
            for prompt, str2 in zip(self.prompt_txt, candidates)
        ]

        # with self.engine.device:
        (rewards,) = self.engine.ancestral(
            context_candidates,
        )

        rewards = np.clip(
            rewards,
            -self.clamp,
            self.clamp,
        ).tolist()

        return rewards


def test_ancestral_simm():
    import re

    def get_last_number(output):

        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return "NaN"

    lora_weight_paths = {
        "gsm8k": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/gsm8k-r16.punica.pt",
        "reward": "/gscratch/ark/graf/cs548/project/punica/model/llama2-reward-gsm8k-r16.punica.pt",
        # "sqlctx": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/sqlctx-r16.punica.pt",
        # "viggo": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/viggo-r16.punica.pt",
    }

    test_prompts = {
        "gsm8k": "<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] Norman High School enrolls an average of 4000 students every year. Butler High School, the neighboring school, enrolls an average of 3/4 as many students as Norman High School. How much greater is the average enrollment at Norman High School than the enrollment at Butler High School? [/INST]\n",
        # "sqlctx": "<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE table_name_2 (popular_votes INTEGER, year VARCHAR, office VARCHAR, percentage VARCHAR)\n<</SYS>>\n[INST] How many votes for the candidate after 1992, 1.59% of the vote, and the office of us representative 4? [/INST]\n",
        # "viggo": "<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] inform(name[The Witcher 3: Wild Hunt], genres[action-adventure, role-playing], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[no]) [/INST]\n",
    }

    def correctness_test(response):
        return get_last_number(response) == "1000"

    model = PunicaSIMM(
        proposal_path=lora_weight_paths["gsm8k"],
        reward_path=lora_weight_paths["reward"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[0],
    )

    outputs = []
    from tqdm import tqdm

    scores = []

    input_ids = [model.tokenizer.encode(test_prompts["gsm8k"])] * 32
    generated_ids, rewards, transition_scores = model.continuation(input_ids)

    correct = []
    texts = []
    for ids in generated_ids:

        # generated_ids, reward, transition_scores = output
        actual_text = model.tokenizer.decode(ids)
        correct.append(correctness_test(actual_text))
        texts.append(actual_text)

    correct = torch.tensor(correct)
    scores = torch.tensor(rewards)

    # correlation between correctness and reward
    print(torch.corrcoef(torch.stack([correct, scores])))

    index = torch.where(correct)[0]

    for i in index:
        print(texts[i])
        print(scores[i])
        print("--" * 40)


def test_ancestral_sism():
    import re

    def get_last_number(output):

        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return "NaN"

    lora_weight_paths = {
        "gsm8k": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/gsm8k-r16.punica.pt",
        "reward": "/gscratch/ark/graf/cs548/project/punica/model/llama2-reward-gsm8k-r16.punica.pt",
        # "sqlctx": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/sqlctx-r16.punica.pt",
        # "viggo": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/viggo-r16.punica.pt",
    }

    test_prompts = {
        "gsm8k": "<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] Norman High School enrolls an average of 4000 students every year. Butler High School, the neighboring school, enrolls an average of 3/4 as many students as Norman High School. How much greater is the average enrollment at Norman High School than the enrollment at Butler High School? [/INST]\n",
        # "sqlctx": "<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE table_name_2 (popular_votes INTEGER, year VARCHAR, office VARCHAR, percentage VARCHAR)\n<</SYS>>\n[INST] How many votes for the candidate after 1992, 1.59% of the vote, and the office of us representative 4? [/INST]\n",
        # "viggo": "<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] inform(name[The Witcher 3: Wild Hunt], genres[action-adventure, role-playing], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[no]) [/INST]\n",
    }

    def correctness_test(response):
        return get_last_number(response) == "1000"

    model = PunicaSISM(
        proposal_path=lora_weight_paths["gsm8k"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[1],
        # dtype=torch.float32,
    )

    outputs = []
    from tqdm import tqdm

    scores = []

    input_ids = [model.tokenizer.encode(test_prompts["gsm8k"])] * 32
    generated_ids, transition_scores = model.continuation(input_ids)

    correct = []
    texts = []
    for ids in generated_ids:

        # generated_ids, reward, transition_scores = output
        actual_text = model.tokenizer.decode(ids)
        correct.append(correctness_test(actual_text))
        texts.append(actual_text)

    correct = torch.tensor(correct)

    for txt in texts:
        print(txt)


def test_ancestral_reward():
    import re

    def get_last_number(output):

        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return "NaN"

    lora_weight_paths = {
        "gsm8k": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/gsm8k-r16.punica.pt",
        "reward": "/gscratch/ark/graf/cs548/project/punica/model/llama2-reward-gsm8k-r16.punica.pt",
        # "sqlctx": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/sqlctx-r16.punica.pt",
        # "viggo": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/viggo-r16.punica.pt",
    }

    test_prompts = {
        "gsm8k": "<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] Norman High School enrolls an average of 4000 students every year. Butler High School, the neighboring school, enrolls an average of 3/4 as many students as Norman High School. How much greater is the average enrollment at Norman High School than the enrollment at Butler High School? [/INST]\n",
        # "sqlctx": "<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE table_name_2 (popular_votes INTEGER, year VARCHAR, office VARCHAR, percentage VARCHAR)\n<</SYS>>\n[INST] How many votes for the candidate after 1992, 1.59% of the vote, and the office of us representative 4? [/INST]\n",
        # "viggo": "<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] inform(name[The Witcher 3: Wild Hunt], genres[action-adventure, role-playing], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[no]) [/INST]\n",
    }

    def correctness_test(response):
        return get_last_number(response) == "1000"

    """model = PunicaR(
        reward_path=lora_weight_paths["reward"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[0],
    )
    """

    # streams = {
    #    device: torch.cuda.Stream(device=device)
    #    for device in [torch.device("cuda:0"), torch.device("cuda:1")]
    # }

    for device_id, device in enumerate(
        [torch.device("cuda:0"), torch.device("cuda:1")]
    ):

        #
        # with torch.cuda.stream(streams[device]):

        reward = ContextualPunicaModel(
            reward_path=lora_weight_paths["reward"],
            model_path="meta-llama/Llama-2-7b-hf",
            max_new_tokens=600,
            max_prompt_length=300,
            device_ids=[device_id],
        )

        data_batch = [{"prompt": test_prompts["gsm8k"]}] * 4

        context = [reward.engine.get_prompt(**data) for data in data_batch]

        reward.set_context(context)

        # with torch.cuda.device(device):
        scores = reward.evaluate(
            ["\n##1000</s>", "\n##1000</s>", "\n##1000</s>", "\n##1000</s>"]
        )

        print(scores)

    # for stream in streams.values():
    #    stream.synchronize()


def test_quest_concurrent():
    import re

    def get_last_number(output):

        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return "NaN"

    lora_weight_paths = {
        "gsm8k": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/gsm8k-r16.punica.pt",
        "reward": "/gscratch/ark/graf/cs548/project/punica/model/llama2-reward-gsm8k-r16.punica.pt",
        # "sqlctx": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/sqlctx-r16.punica.pt",
        # "viggo": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/viggo-r16.punica.pt",
    }

    test_prompts = {
        "gsm8k": "<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] Norman High School enrolls an average of 4000 students every year. Butler High School, the neighboring school, enrolls an average of 3/4 as many students as Norman High School. How much greater is the average enrollment at Norman High School than the enrollment at Butler High School? [/INST]\n",
        # "sqlctx": "<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE table_name_2 (popular_votes INTEGER, year VARCHAR, office VARCHAR, percentage VARCHAR)\n<</SYS>>\n[INST] How many votes for the candidate after 1992, 1.59% of the vote, and the office of us representative 4? [/INST]\n",
        # "viggo": "<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] inform(name[The Witcher 3: Wild Hunt], genres[action-adventure, role-playing], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[no]) [/INST]\n",
    }

    def correctness_test(response):
        return get_last_number(response) == "1000"

    model = PunicaSIMM(
        proposal_path=lora_weight_paths["gsm8k"],
        reward_path=lora_weight_paths["reward"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[1],
    )

    proposal = JointRLHFSuffixProposal(
        model=model,
    )

    chain = Quest(
        input_data=[{"prompt": test_prompts["gsm8k"]}] * 32,
        proposal=proposal,
        beta=1.0,
    )

    results = chain.run(
        steps=10,
        use_tqdm=True,
    )


def test_quest_sequential():
    import re

    def get_last_number(output):

        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return "NaN"

    lora_weight_paths = {
        "gsm8k": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/gsm8k-r16.punica.pt",
        "reward": "/gscratch/ark/graf/cs548/project/punica/model/llama2-reward-gsm8k-r16.punica.pt",
        # "sqlctx": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/sqlctx-r16.punica.pt",
        # "viggo": "/mmfs1/gscratch/ark/graf/cs548/project/punica/model/viggo-r16.punica.pt",
    }

    test_prompts = {
        "gsm8k": "<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] Norman High School enrolls an average of 4000 students every year. Butler High School, the neighboring school, enrolls an average of 3/4 as many students as Norman High School. How much greater is the average enrollment at Norman High School than the enrollment at Butler High School? [/INST]\n",
        # "sqlctx": "<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE table_name_2 (popular_votes INTEGER, year VARCHAR, office VARCHAR, percentage VARCHAR)\n<</SYS>>\n[INST] How many votes for the candidate after 1992, 1.59% of the vote, and the office of us representative 4? [/INST]\n",
        # "viggo": "<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] inform(name[The Witcher 3: Wild Hunt], genres[action-adventure, role-playing], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[no]) [/INST]\n",
    }

    def correctness_test(response):
        return get_last_number(response) == "1000"

    model = PunicaSISM(
        proposal_path=lora_weight_paths["gsm8k"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[0],
    )

    reward = ContextualPunicaModel(
        reward_path=lora_weight_paths["reward"],
        model_path="meta-llama/Llama-2-7b-hf",
        max_new_tokens=600,
        max_prompt_length=300,
        device_ids=[1],
    )

    data_batch = [{"prompt": test_prompts["gsm8k"]}] * 4

    context = [model.get_prompt(**data) for data in data_batch]

    reward.set_context(context)

    proposal = RLHFSuffixProposal(
        model=model,
        reward=reward,
    )

    chain = Quest(
        input_data=data_batch,
        proposal=proposal,
        beta=1.0,
    )

    results = chain.run(
        steps=2,
        use_tqdm=True,
    )


if __name__ == "__main__":
    # test_ancestral_sism()
    # test_ancestral_simm()
    # test_ancestral_reward()
    # test_quest_sequential()
    test_quest_concurrent()
