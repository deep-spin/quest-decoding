
from quest.reward.model import RewardModel


def integrated_test():
    
    
    rm = RewardModel(
       "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    
    
    print(
        rm.evaluate(["I love this movie", "I hate this movie"])
    )
    
    pass
    


integrated_test()

print("passed all tests")
