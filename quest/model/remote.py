from quest.model.base import (
    LocalLanguageModel,
    DEFAULT_TEMPLATE,
)


from quest.utils.list import (
    flatten_list,
    unflatten_list,
)
from typing import List, Optional, Any
import requests
from langchain.prompts import PromptTemplate
from quest.model.base import LocalLanguageModel
import numpy as np
import json
import time


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

        try:
            self.server_url = registry.get(model_path).rstrip("/")
        except:
            self.server_url = "http://localhost:8000"

        self.timeout = timeout
        self.session = requests.Session()
        self.max_retries = max_retries
        self.registry = registry

        # Test connection and get model info
        self._check_health()

    def _check_health(self):
        """Check if the server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    def continuation(self, prompt, prefix=None):

        if prefix is None:
            input_data = prompt
        else:
            input_data = [x[0] + x[1] for x in zip(prompt, prefix)]

            # print(input_data)

        prompt_text = self.decode_tokenize(
            input_data, include_special_tokens=True, spaces_between_special_tokens=False
        )

        print("--" * 20)
        print(max(map(len, input_data)))

        len_of_min = min(map(len, input_data))
        discount = min(self.max_prompt_length - len_of_min, 0)
        # print(prompt_text)
        # Prepare request payload
        payload = {
            "model": self.model_path,
            "prompt": prompt_text,
            "temperature": self.temperature,
            "logprobs": 1,
            "max_tokens": self.max_new_tokens - discount,
            "stop": self.stop_tokens,
            # "include_stop_str_in_output": True,
            # "skip_special_tokens": self.skip_special_tokens,
            # "spaces_between_special_tokens": False,
        }

        # Make request to vLLM server
        response = self._make_request(endpoint="v1/completions", payload=payload)
        # response = requests.post(f"{self.server_url}/generate", json=payload)

        """
        outputs = response["outputs"]
        # Extract completions and scores
        completion = [list(out["token_ids"]) for out in outputs]
        scores = [[logprob for logprob in out["logprobs"]] for out in outputs]
        """
        # Get the choices from the response
        outputs = response["choices"]

        # Extract completions and logprobs
        completions = []
        logprobs = []
        scores = []

        # space_token = "Ġ"

        for choice in outputs:

            if "logprobs" in choice:
                # Extract token ids if available

                completions.append(
                    choice["text"] + self.tokenizer.eos_token
                )  # fallback to text if no token_ids

                # Extract the log probabilities
                logprobs.append(choice["logprobs"]["token_logprobs"])

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
        attempt = 0
        last_exception = None

        # Prepare prompts
        prompts = [self.get_prompt(**data) for data in input_data] * n

        payload = {
            "model": self.model_path,
            "prompt": prompts,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            # "top_p": top_p,
            "n": 1,  # We handle multiple generations through prompt repetition
            "stop": self.stop_tokens,
            # "best_of": best_of if best_of is not None else 1,
        }

        result = self._make_request("v1/completions", payload)

        completions = [choice["text"] for choice in result["choices"]]
        # Reshape the completions according to n
        return unflatten_list(completions, [n] * len(input_data))

    def _make_request(self, endpoint: str, payload: dict):
        """Generic request handler with retry logic"""
        attempt = 0
        last_exception = None
        self.server_url = self.registry.get(self.model_path).rstrip("/")

        while attempt < self.max_retries:
            try:
                response = self.session.post(
                    f"{self.server_url}/{endpoint}",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ConnectionError) as e:
                attempt += 1
                last_exception = e
                self.server_url = self.registry.get(self.model_path).rstrip("/")

                print(f"Attempt {attempt} failed: {str(e)}->")
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Failed to complete request after {self.max_retries} attempts. "
                        f"Last error: {str(last_exception)}"
                    ) from last_exception

                # Exponential backoff
                time.sleep(min(2**attempt, 60))
                continue

        raise RuntimeError("Unexpected end of retry loop")
