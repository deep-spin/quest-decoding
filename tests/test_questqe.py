from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
from quest.decoding import Quest
from quest.index import Uniform
from quest.reward.qe import QEModel
import os


def integrated_test():

    template = PromptTemplate.from_template(
        "Translate this from {source_language} to {target_language}:\n{source_language}: {source_sentence}\n{target_language}:"
    )

    test_input_data = [
        {
            "source_language": "English",
            "target_language": "French",
            "source_sentence": "Hello, how are you?",
        }
    ]

    sources = [data["source_sentence"] for data in test_input_data]

    model = VLLM(
        model_path="haoranxu/ALMA-7B",
        prompt_template=template,
        download_dir=os.environ["HF_HOME"],
        gpu_memory_utilization=0.6,
    )

    reward = QEModel("Unbabel/wmt23-cometkiwi-da")  # sentiment model.
    reward.set_sources(sources)  # QE model requires sources to be set.

    index = Uniform()

    chain = Quest(
        input_data=test_input_data,
        model=model,
        reward=reward,
        dist=index,
    )

    chain_outputs = chain.run(
        steps=10,
        use_tqdm=True,
    )

    print(chain_outputs.samples)


integrated_test()

print("passed all tests")
