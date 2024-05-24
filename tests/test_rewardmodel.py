
from quest.reward.model import RewardModel


def integrated_test():
    
    
    rm = RewardModel(
        "lvwerra/distilbert-imdb"
    )
    
    
    print(
        rm.evaluate(["I love this movie", "I hate this movie"])
    )
    
    pass
    


integrated_test()

print("passed all tests")
