import requests
import time
from typing import List, Optional, Union

from quest.reward.base import Reward
from tqdm import tqdm
from literegistry import RegistryHTTPClient
import asyncio

from quest.utils.list import split_into_groups


class RemoteReward(Reward):
    def __init__(
        self,
        model_path: str,
        registry,
        max_retries: int = 50,
        polling_interval: float = 0.5,
        timeout: float = 300,  # 5 minutes default timeout
        reward_type: str = "contextual",
        batch_size=32,
        max_parallel_requests=32,
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
            results = await client.post("evaluate", payload)

        rewards = [r for result in results for r in result["rewards"]]

        return rewards

    """ try:
                    # Submit evaluation request
                    payload = {"texts": candidates}
                    if self._context is not None:
                        payload["context"] = self._context

                    response = self.session.post(f"{self.base_url}/evaluate", json=payload)
                    response.raise_for_status()
                    task_id = response.json()["task_id"]

                    # Poll for results with retry logic
                    while time.time() - start_time < self.timeout:
                        try:
                            response = self.session.get(f"{self.base_url}/task/{task_id}")
                            response.raise_for_status()
                            result = response.json()

                            if result["status"] == "completed":
                                return result["rewards"]
                            elif result["status"] == "failed":
                                raise RuntimeError(
                                    f"Task failed: {result.get('error', 'Unknown error')}"
                                )

                            time.sleep(self.polling_interval)

                        except (requests.RequestException, ConnectionError) as e:
                            # Connection error during polling - try new server
                            self.base_url = self.get_new_url()
                            last_exception = e
                            continue

                    raise TimeoutError(f"Evaluation timed out after {self.timeout} seconds")

                except (requests.RequestException, ConnectionError) as e:
                    # Connection error during submission - try new server
                    attempt += 1
                    last_exception = e
                    self.base_url = self.get_new_url()

                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"Failed to evaluate after {self.max_retries} attempts. "
                            f"Last error: {str(last_exception)}"
                        ) from last_exception

                    # Add exponential backoff between retries
                    time.sleep(min(2**attempt, 30))  # Cap at 30 seconds
                    continue

            # This should never be reached due to the raise in the loop above
            raise RuntimeError("Unexpected end of retry loop")
    """

    def evaluate(self, candidates, use_tqdm=True, **kwargs):

        # break candidates into batches
        batches_data = [
            candidates[i : i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        context_batches = [
            self._context[i : i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        batches = zip(
            batches_data,
            context_batches,
        )

        if use_tqdm:
            batches = tqdm(batches, desc="Evaluating", total=len(batches_data))

        results = []
        for batch, context in batches:
            # self.set_context(context)
            # results.extend(self._evaluate(batch))

            individual = [{"texts": t, "context": c} for t, c in zip(batch, context)]

            payload = [
                {
                    "texts": [p["texts"] for p in packed],
                    "context": [p["context"] for p in packed],
                }
                for packed in split_into_groups(individual, self.max_parallel_requests)
            ]

            results.extend(asyncio.run(self._evaluate(payload)))

            # print(f"Evaluated {len(results)} samples")

        return results
