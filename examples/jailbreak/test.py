from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
from quest import Quest, RLHFSuffixProposal

from quest.reward.base import (
    Reward,
    ConstantReward,
    BackwardReward,
)

from quest.index import Uniform
from quest.reward.model import (
    ContextualRewardModel,
)
from datasets import load_dataset
import os

dataset_path = "Anthropic/hh-rlhf"


def process_data(entry):
    breaks = entry["chosen"].split("\n\n")

    twopb = breaks[-1].split(":")
    descriptor = twopb[0]
    answer = ":".join(twopb[1:])

    breaks[-1] = f"{descriptor}: "
    prompt = "\n\n".join(breaks)

    return {
        "prompt": prompt,
        "answer": answer,
    }


def main(
    oposite: bool = False,
    beta: float = 0.5,
    steps: int = 25,
    temperature: float = 0.6,
    n: int = 1,
    model_path: str = "meta-llama/Meta-Llama-3-8B",
    reward_model_path: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
):

    ds = load_dataset(
        dataset_path, split="test"
    )

    psd = ds.map(process_data)

    data_iterable = list(psd)[:n]

    model = VLLM(
        model_path=model_path,
        # prompt_template=template,
        download_dir=os.environ.get(
            "HF_HOME", "/tmp/"
        ),
        stop_tokens=["\n"],
        temperature=temperature,
    )

    reward = ContextualRewardModel(
        model_path=reward_model_path
    )  # sentiment model.
    # ConstantReward(1.0)#

    context = [
        model.get_prompt(**data)
        for data in data_iterable
    ]

    reward.set_context(context)

    if oposite:
        reward = BackwardReward(reward)

    index = Uniform()

    chain = Quest(
        input_data=data_iterable,
        proposal=RLHFSuffixProposal(
            model=model, dist=index
        ),
        reward=reward,
        beta=beta,
    )

    chain_outputs = chain.run(
        steps=steps,
        use_tqdm=True,
    )

    print(data_iterable)

    for s in chain_outputs.state_path:
        print((s["reward"], s["text"]))


if __name__ == "__main__":

    import fire

    fire.Fire(main)
