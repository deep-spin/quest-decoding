from quest.reward.mt import QEModel


def integrated_test():

    model = QEModel()
    model.set_sources(
        [
            "This is a test.",
            "This is another test.",
        ]
    )

    rewards = model.evaluate(
        [
            "Isto é um test.",
            "Isto é outro test.",
        ]
    )

    print(rewards)


integrated_test()
print("passed all tests")
