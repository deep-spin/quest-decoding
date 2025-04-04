from quest.reward.remote import RemoteReward
from literegistry import RegistryClient, FileSystemKVStore


def test(
    model_path: str = "/gscratch/ark/graf/LLaMA-Factory/saves/llama3/8b/full/reward/",
    reward_type="value",
):

    registry = RegistryClient(
        store=FileSystemKVStore("/gscratch/ark/graf/registry"),
        max_history=3600,
        cache_ttl=60,
        service_type="model_path",
    )

    client = RemoteReward(
        model_path=model_path,
        registry=registry,
        reward_type=reward_type,
    )

    # client.base_url = base_url

    client.set_context(
        [
            "This is a test context.",
            "This is another test context.",
        ]
        * 1
    )
    # Single batch evaluation
    rewards = client.evaluate(["Text 1", "Text 2"] * 1)
    print(rewards)


# Example usage
if __name__ == "__main__":
    import fire

    fire.Fire(test)
