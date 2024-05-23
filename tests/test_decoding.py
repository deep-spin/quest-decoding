from langchain.prompts import PromptTemplate
from quest.model.vllm import VLLM
from quest.decoding import Quest, QuestRLHF
from quest.reward.base import Reward
from quest.index import Uniform

import os



class LengthReward(Reward):

    def __init__(
        self,
        modes=[ ((0, 10),0), ((30, 40),0)],
        default_value=-4,
    ):
        super().__init__()
        self.modes = modes
        self.default_value=default_value

    def get_lengths(self, candidates):

        lengths =(
            list(map(lambda x: len((x).split()), candidates))
        )

        return lengths

   
    def evaluate(self, candidates, **kwargs):
        lengths = self.get_lengths(candidates)

        rewards=[]

        for ln in lengths:
            for m,value in self.modes:
                reward = self.default_value 
                if (ln >= m[0]) and (ln <= m[1]):
                    reward = value

            rewards.append(reward)
        
        return rewards

    


def integrated_test():
    
    template =  PromptTemplate.from_template(
        "Translate this from {source_language} to {target_language}:\n{source_language}: {source_sentence}\n{target_language}:"
    )
    
    test_input_data = [{
        "source_language": "English",
        "target_language": "French",
        "source_sentence": "Hello, how are you?"
    }]

    model = VLLM(
        model_path="haoranxu/ALMA-7B",
        prompt_template=template,
        download_dir=os.environ["HF_HOME"],
    )

    reward = LengthReward() 
    
    index = Uniform()
    
    chain = QuestRLHF(
        input_data=test_input_data,
        model=model,
        reward=reward,
        dist=index,   
    )
    
    chain_outputs = chain.run(
        steps=10,
        use_tqdm=True,
    )
    
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