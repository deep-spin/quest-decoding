from quest.model.base import (
    LocalLanguageModel,
    DEFAULT_TEMPLATE,
)


from quest.utils.list import flatten_list, unflatten_list, split_into_groups
from quest.utils.http import async_wrapper
from typing import List, Optional, Any
import requests
from langchain.prompts import PromptTemplate
from quest.model.base import LocalLanguageModel
import numpy as np
import json
import time
from literegistry import RegistryHTTPClient
import asyncio


class RemoteVLLM(LocalLanguageModel):
    def __init__(
        self,
        registry,
        model_path: str,  # For compatibility, not used
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens: int = 600,
        max_prompt_length: int = 300,
        stop_tokens: List[str] = ["</s>"],
        temperature: float = 1.0,
        skip_special_tokens: bool = False,
        timeout: float = 300,
        max_retries: int = 50,
        batch_size=32,
        **llm_kwargs,
    ):
        super().__init__(
            model_path=model_path,
            prompt_template=prompt_template,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            stop_tokens=stop_tokens,
            skip_special_tokens=skip_special_tokens,
        )

        self.http_client = RegistryHTTPClient(
            registry=registry,
            value=model_path,
            max_parallel_requests=batch_size,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.value = model_path
        self.batch_size = batch_size

        self.registry = registry

        # Test connection and get model info
        asyncio.run(self._check_health())

    async def _check_health(self):
        """Check if the server is healthy."""
        try:

            async with self.http_client as client:
                # server_url = self.registry.get(self.model_path).rstrip("/")
                response = await client.get("v1/models")
                # response = self.session.get(f"{self.server_url}/health")
                # response.raise_for_status()

            print(f"Server {self.value} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    async def _check_health(self):
        """Check if the server is healthy."""
        try:

            async with self.http_client as client:
                # server_url = self.registry.get(self.model_path).rstrip("/")
                response = await client.get("v1/models")
                # response = self.session.get(f"{self.server_url}/health")
                # response.raise_for_status()

            print(f"Server {self.value} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    def continuation(self, prompt, prefix=None):

        if prefix is None:
            input_data = prompt
        else:
            input_data = [x[0] + x[1] for x in zip(prompt, prefix)]

            ninput = []
            for x in input_data:
                nx = len(x)
                if nx > self.max_prompt_length:
                    ninput.append(x[nx - self.max_prompt_length :])
                else:
                    ninput.append(x)

            input_data = ninput

        prompt_text = self.decode_tokenize(
            input_data, skip_special_tokens=False, spaces_between_special_tokens=False
        )

        # print(prompt_text)
        # Prepare request payload
        payload = [
            {
                "model": self.model_path,
                "prompt": p,
                "temperature": self.temperature,
                "logprobs": 1,
                "max_tokens": self.max_new_tokens,
                "stop": self.stop_tokens,
                # "include_stop_str_in_output": True,
                # "skip_special_tokens": self.skip_special_tokens,
                # "spaces_between_special_tokens": False,
            }
            for p in split_into_groups(prompt_text, self.batch_size)
        ]

        # Make request to vLLM server
        results = asyncio.run(
            self._make_request(endpoint="v1/completions", payload=payload)
        )
        # response = requests.post(f"{self.server_url}/generate", json=payload)

        completions = [
            choice["text"] + self.tokenizer.eos_token
            for result in results
            for choice in result["choices"]
        ]

        logprobs = [
            choice["logprobs"]["token_logprobs"]
            for result in results
            for choice in result["choices"]
        ]

        """
        outputs = response["outputs"]
        # Extract completions and scores
        completion = [list(out["token_ids"]) for out in outputs]
        scores = [[logprob for logprob in out["logprobs"]] for out in outputs]
        """
        # Get the choices from the response
        # outputs = response["choices"]

        completion_ids = [xi[1:] for xi in self.tokenize(completions)]

        return completion_ids, logprobs

    def ancestral(
        self,
        input_data,
        top_p: float = 1.0,
        min_p: float = 0.0,
        use_beam_search: bool = False,
        best_of: Optional[int] = None,
        n: int = 1,
    ):
        """
        Generate completions using ancestral sampling via OpenAI API format with retry logic.

        Args:
            input_data: Input data for generations
            top_p: Top-p sampling parameter
            min_p: Minimum probability threshold
            use_beam_search: Whether to use beam search
            best_of: Number of candidates to generate per prompt
            n: Number of generations per input
            max_retries: Maximum number of retry attempts

        Returns:
            List of completions

        Raises:
            RuntimeError: If all retry attempts fail
        """

        # Prepare prompts
        prompts = [self.get_prompt(**data) for data in input_data] * n

        payload = [
            {
                "model": self.model_path,
                "prompt": p,  # "say hi",  # prompts,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                # "top_p": top_p,
                "n": 1,  # We handle multiple generations through prompt repetition
                # "stop": self.stop_tokens,
                # "best_of": best_of if best_of is not None else 1,
            }
            for p in split_into_groups(prompts, self.batch_size)
        ]

        results = asyncio.run(self._make_request("v1/completions", payload))

        completions = [
            choice["text"] for result in results for choice in result["choices"]
        ]
        # Reshape the completions according to n
        return unflatten_list(completions, [n] * len(input_data))

    async def _make_request(self, endpoint: str, payload: dict):

        async with self.http_client as client:
            # server_url = self.registry.get(self.model_path).rstrip("/")
            response = await client.post("v1/completions", payload)
            # response = self.session.get(f"{self.server_url}/health")
            # response.raise_for_status()

        return response
