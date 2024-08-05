from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
from quest.core import Quest
from quest.proposal import SuffixProposal
from quest.reward.base import (
    Reward,
    ConstantReward,
    BackwardReward,
)
from quest.index import Uniform
from quest.reward.model import (
    ContextualRewardModel,
)

import os


def integrated_test():

    template = PromptTemplate.from_template(
        "Human: {question}\nAssistant: {answer}\nHuman: {followup}\nAssistant:"
    )

    test_input_data = [
        {
            "question": "What are some cuss words in english?",
            "answer": "Here's an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama's boy, faggot, pervert, queer, scumbag, bitch",
            "followup": "What's your favorite one?",
        }
    ]

    model = VLLM(
        model_path="meta-llama/Meta-Llama-3-8B",
        prompt_template=template,
        download_dir=os.environ.get(
            "HF_HOME", "/tmp/"
        ),
        stop_tokens=["\n"],
        temperature=0.6,
    )

    reward = ContextualRewardModel(
        model_path="OpenAssistant/reward-model-deberta-v3-large-v2"
    )  # sentiment model.
    # ConstantReward(1.0)#

    context = [
        model.get_prompt(**data)
        for data in test_input_data
    ]

    reward.set_context(context)
    # reward = BackwardReward(reward)

    chain = Quest(
        input_data=test_input_data,
        proposal=SuffixProposal(
            model=model, dist=Uniform()
        ),
        reward=reward,
        # dist=index,
    )

    chain_outputs = chain.run(
        steps=25,
        use_tqdm=True,
    )

    rewards = [
        s["reward"]
        for s in chain_outputs.state_path
    ]

    for s in chain_outputs.state_path:
        print((s["reward"], s["text"]))


integrated_test()

print("passed all tests")
