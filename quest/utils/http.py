import asyncio
import aiohttp
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from collections import defaultdict


class SentenceTooLongError(Exception):
    """
    Custom exception for when a sentence exceeds the maximum allowed length in VLLM.
    Includes the sentence length and maximum allowed length for context.
    """

    def __init__(self, sentence_length, max_length=None, original_error=None):
        self.sentence_length = sentence_length
        self.max_length = max_length
        self.original_error = original_error
        message = f"Sentence length ({sentence_length} tokens) exceeded maximum allowed length"
        if max_length:
            message += f" of {max_length} tokens"
        if original_error:
            message += f". Original error: {str(original_error)}"
        super().__init__(message)


def async_wrapper(fx):
    import asyncio

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(fx)
    return response


async def make_http_request(
    session: aiohttp.ClientSession,
    server: str,
    endpoint: str,
    payload: Dict,
    timeout: float,
) -> Dict:
    """Make a single HTTP POST request to a server."""
    async with session.post(
        f"{server.rstrip('/')}/{endpoint}",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:

        if response.status == 400:
            error_json = await response.json()  # For requests library
            error_msg = error_json.get("error", {}).get("message", "")
            if "Bad Request" in error_msg:
                print(f"Bad Request error: {error_msg}")
                # You can handle the error here (e.g., truncate or retry)
                raise SentenceTooLongError(f"Bad Request detected: {error_msg}")

        response.raise_for_status()
        return await response.json()


async def request_with_server_rotation(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: Dict,
    timeout: float,
    max_retries: int,
    initial_server_idx: int = 0,
    fetch=None,
    report=None,
) -> Tuple[Dict, int]:
    """Make a request with automatic server rotation on failure."""
    servers = fetch()
    total_servers = len(servers)

    attempt = 0
    server_idx = initial_server_idx

    while attempt < max_retries:
        start_time = asyncio.get_event_loop().time()

        try:
            server_idx = server_idx % total_servers
            server = servers[server_idx].rstrip("/")
            result = await make_http_request(
                session, server, endpoint, payload, timeout
            )

            if report:
                report(server, asyncio.get_event_loop().time() - start_time)

            return result, server_idx

        # except SentenceTooLongError as e:
        #    print("Size issue :", e)

        # new_payload = payload.copy()
        # payload["prompt"] = payload["prompt"][:5000]

        except Exception as e:
            print(f"Attempt {attempt + 1} failed on server {server}: {str(e)}")
            attempt += 1
            # Fetch updated server list if provided

            if report:
                report(server, asyncio.get_event_loop().time() - start_time)

            if fetch:
                servers = fetch(force=True)
                total_servers = len(servers)

            # Rotate to next server
            server_idx = server_idx + 1

            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed after {max_retries} attempts across {total_servers} servers: {str(e)}"
                )
            await asyncio.sleep(min(2**attempt, 60))

    raise RuntimeError("Unexpected end of retry loop")


async def make_parallel_requests(
    session: aiohttp.ClientSession,
    endpoint: str,
    payloads: List[Dict],
    timeout: float,
    max_retries: int,
    max_parallel_requests: int,
    fetch=None,
    report=None,
) -> List[Dict]:
    """Make multiple requests in parallel with server rotation."""

    # Initialize tasks with different starting servers
    tasks = []
    for i, payload in enumerate(payloads):
        tasks.append(
            request_with_server_rotation(
                session,
                endpoint,
                payload,
                timeout,
                max_retries,
                initial_server_idx=i,
                fetch=fetch,
                report=report,
            )
        )

    # Use semaphore to limit concurrent requests
    sem = asyncio.Semaphore(max_parallel_requests)

    async def bounded_request(task):
        async with sem:
            return await task

    # Gather results with bounded concurrency
    bounded_tasks = [bounded_request(task) for task in tasks]
    results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

    # Check for exceptions and extract results
    final_results = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        response, _ = result  # Unpack the response and server_idx
        final_results.append(response)

    return final_results
