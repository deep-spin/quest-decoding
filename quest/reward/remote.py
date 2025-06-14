import requests
import time
from typing import List, Optional, Union

from quest.reward.base import Reward
from tqdm import tqdm
from literegistry import RegistryHTTPClient
import asyncio

from quest.utils.list import split_into_groups, chunked


class RemoteReward(Reward):
    def __init__(
        self,
        model_path: str,
        registry,
        max_retries: int = 50,
        polling_interval: float = 0.5,
        timeout: float = 300,  # 5 minutes default timeout
        reward_type: str = "contextual",
        batch_size=64,
        max_parallel_requests=64,
    ):
        """
        Client for interacting with the Reward Model Server.

        Args:
            base_url: Base URL of the server
            max_retries: Maximum number of status check retries
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait for result in seconds
            context: Optional default context to use for evaluations
        """
        """
        Client for interacting with the Reward Model Server.
        
        Args:
            base_url: Base URL of the server
            max_retries: Maximum number of status check retries
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait for result in seconds
        """

        if reward_type == "contextual":
            super().__init__(f"crm:{model_path}")
        elif reward_type == "value":
            super().__init__(f"vh:{model_path}")
        elif reward_type == "qe":
            super().__init__(f"qe:{model_path}")
        else:
            super().__init__(f"rm:{model_path}")

        self.max_retries = max_retries
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.http_client = RegistryHTTPClient(
            registry=registry,
            value=model_path,
            max_parallel_requests=max_parallel_requests,
            timeout=timeout,
            max_retries=max_retries,
        )
        # self.session = requests.Session()
        self._context = None  # Store context as instance variable
        self.registry = registry
        self.model_path = model_path
        # self.base_url = self.get_new_url()
        self.batch_size = batch_size
        self.max_parallel_requests = max_parallel_requests
        # Test connection and get model info
        asyncio.run(self._check_health())

    async def _check_health(self):
        """Check if the server is healthy."""
        try:

            async with self.http_client as client:
                # server_url = self.registry.get(self.model_path).rstrip("/")
                response = await client.get("health")
                # response = self.session.get(f"{self.server_url}/health")
                # response.raise_for_status()

            print(f"Server {self.model_path} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    def set_context(self, context: List[str]):
        """Set the default context for all evaluations."""
        self._context = context

    def get_context(self) -> Optional[List[str]]:
        """Get the current default context."""
        return self._context

    async def _evaluate(
        self,
        payload,
    ) -> List[float]:
        """
        Submit texts for evaluation with retry logic.

        Args:
            texts: List of texts to evaluate
            use_tqdm: Whether to use progress bar
            max_retries: Maximum number of retry attempts before giving up

        Returns:
            List of reward scores

        Raises:
            RuntimeError: If all retry attempts fail
            TimeoutError: If evaluation times out
        """
        async with self.http_client as client:
            results = await client.post("evaluate", payload, track=False)

        rewards = [r for result in results for r in result["rewards"]]

        return rewards

    def evaluate(self, candidates, use_tqdm=False, **kwargs):

        num_servers = len(asyncio.run(self.registry.get_all(self.model_path)))

        temp_batch = min(
            max((len(candidates)) // max(num_servers, 1), 32), self.batch_size
        )

        # print("reward_batch", temp_batch)

        payloads = [
            {"texts": t, "context": c} for t, c in zip(candidates, self._context)
        ]

        packed_payload = [
            {
                "texts": [p["texts"] for p in packed],
                "context": [p["context"] for p in packed],
            }
            for packed in chunked(payloads, temp_batch)
        ]

        results = asyncio.run(self._evaluate(packed_payload))

        return results
