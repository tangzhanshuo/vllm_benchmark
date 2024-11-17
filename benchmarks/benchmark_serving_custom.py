r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import cohere
import io
import json
import os
import random
import time
import torch
import warnings
from dataclasses import dataclass
from datasets import load_from_disk
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func_custom import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func_custom import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[List[Dict]]:
    
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data["conversations"] for data in dataset if len(data["conversations"]) >= 2]

    if len(dataset) <= num_requests:
        random.shuffle(dataset)
        return dataset
    
    priority_factor = 0.5
    weights = [len(data) ** priority_factor for data in dataset]
    total = sum(weights)
    weights = [w/total for w in weights]

    filtered_dataset: List[Dict] = []
    for i in range(num_requests):
        choice = random.choices(dataset, weights=weights, k=1)[0]
        filtered_dataset.append(choice)
        idx = dataset.index(choice)
        dataset.pop(idx)
        weights.pop(idx)

    return filtered_dataset


def generate_requests(input_requests: List[List[Dict]], tokenizer: PreTrainedTokenizerBase) -> List[List[RequestFuncInput]]:
    processed_requests = []

    for conversation in input_requests:
        processed_conversation = []
        content = []

        for i, turn in enumerate(conversation):
            processed_turn = {}
            processed_turn["content"] = [{"type": "text", "text": turn["value"]}]

            if turn["from"] == "gpt":
                processed_turn["role"] = "assistant"
            elif turn["from"] == "human":
                processed_turn["role"] = "user"
            else:
                raise ValueError(f"Invalid 'from' field: {turn['from']}")

            content.append(processed_turn)

            if processed_turn["role"] == "user" and i != len(conversation) - 1 and conversation[i+1]["from"] == "gpt":
                req = RequestFuncInput()
                req.content = content.copy()
                req.output_len = len(tokenizer.encode(conversation[i+1]["value"]))
                processed_conversation.append(req)

        processed_requests.append(processed_conversation)

    return processed_requests


def get_relevant_docs_batch(query_embeddings, docs, doc_embeddings, top_k):
    # Compute similarity scores for all queries at once
    dot_scores = torch.mm(query_embeddings, doc_embeddings.transpose(0, 1))
    top_k_results = torch.topk(dot_scores, k=min(top_k, dot_scores.size(1)))
    
    # Get top-k documents for each query
    all_relevant_docs = []
    for indices in top_k_results.indices:
        relevant_docs = []
        for doc_id in indices.tolist():
            relevant_docs.append({
                'title': docs[doc_id].get('title', ''),
                'text': docs[doc_id].get('text', '')
            })
        all_relevant_docs.append(relevant_docs)
    
    return all_relevant_docs


def get_query_string(request: RequestFuncInput) -> str:
    return request.content[-1]["content"][0]["text"]


def generate_rag(input_requests: List[List[RequestFuncInput]], dataset: str, model: str, top_k: int) -> List[List[RequestFuncInput]]:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))

    docs = load_from_disk(dataset)
    doc_embeddings = torch.tensor(docs['emb'])

    # Gather all queries and their locations
    queries = []
    query_locations = []  # Store (conv_idx, turn_idx) for each query
    
    for conv_idx, conversation in enumerate(input_requests):
        for turn_idx, turn in enumerate(conversation):
            query = get_query_string(turn)  # External function call
            if query:  # Only process if we get a valid query
                queries.append(query)
                query_locations.append((conv_idx, turn_idx))
    
    # If no queries to process, return the original input
    if not queries:
        return input_requests
    
    # Get embeddings for all queries in one batch
    response = co.embed(texts=queries, model=model)
    query_embeddings = torch.tensor(response.embeddings)
    
    # Get relevant documents for all queries
    all_relevant_docs = get_relevant_docs_batch(
        query_embeddings,
        docs,
        doc_embeddings,
        top_k
    )
    
    # Create a copy of input_requests to modify
    result_requests = [[turn for turn in conv] for conv in input_requests]
    
    # Update conversations with relevant documents
    for (conv_idx, turn_idx), relevant_docs in zip(query_locations, all_relevant_docs):
        # Build context string
        docs = "Relevant context:\n"
        for doc in relevant_docs:
            docs += f"Title: {doc['title']}\n"
            docs += f"Content: {doc['text']}\n\n"
        
        # Update the result
        result_requests[conv_idx][turn_idx].content[-1]["content"].insert(0, {"type": "text", "text": docs})
    
    return result_requests


async def get_request(
    input_requests: List[List[RequestFuncInput]],
    tokenizer: PreTrainedTokenizerBase,
    request_rate: float,
    burstiness: float,
    turn_request_rate: float,
    turn_burstiness: float
) -> AsyncGenerator[RequestFuncInput, None]:
    """
    Asynchronously generates requests at a specified rate with two-level burstiness:
    one for conversations and one for turns within conversations.
    
    Args:
        input_requests: List of conversations, each containing multiple turns.
        request_rate: Rate at which conversations are initiated (conversations/s).
        conversation_burstiness: Burstiness factor for conversation initiation.
        turn_burstiness: Burstiness factor for turns within a conversation.
        mean_turn_interval: Mean interval between turns in a conversation.
    """
    
    # Calculate scale parameter for conversation intervals
    assert burstiness > 0, f"Invalid conversation burstiness: {burstiness}"
    assert turn_burstiness > 0, f"Invalid turn burstiness: {turn_burstiness}"
    
    conversation_theta = 1.0 / (request_rate * burstiness)
    turn_theta = 1.0 / (turn_request_rate * turn_burstiness)
    
    if request_rate == float("inf"):
        # All conversations start at time 0
        conversation_start_times = [0.0] * len(input_requests)
    else:
        # Generate start times for conversations
        intervals = np.random.gamma(
            shape=burstiness,
            scale=conversation_theta,
            size=len(input_requests)
        )
        conversation_start_times = np.cumsum(intervals)
    
    # Generate all turn timings and requests in advance
    all_requests = []
    for conv_idx, conversation in enumerate(input_requests):
        conv_start = conversation_start_times[conv_idx]
        
        # Generate intervals between turns
        if len(conversation) > 1:
            turn_intervals = np.random.gamma(
                shape=turn_burstiness,
                scale=turn_theta,
                size=len(conversation) - 1
            )
            turn_times = np.concatenate([[0], np.cumsum(turn_intervals)])
        else:
            turn_times = [0]
            
        # Add conversation start time to turn times
        turn_times = [conv_start + t for t in turn_times]
        
        # Create (time, request) pairs
        for turn_idx, request in enumerate(conversation):
            all_requests.append((turn_times[turn_idx], request))
    
    # Sort requests by time
    all_requests.sort(key=lambda x: x[0])
    
    # Yield requests with appropriate delays
    last_time = 0.0
    for request_time, request in all_requests:
        wait_time = request_time - last_time
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        yield request
        last_time = request_time


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            # output_len = len(
            #     tokenizer(outputs[i].generated_text,
            #               add_special_tokens=False).input_ids)
            output_len = outputs[i].output_len
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len
            tpot = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len -
                                                                 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if gootput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in gootput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(gootput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in gootput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(gootput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in gootput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(gootput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[List[Dict]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    turn_request_rate: float,
    turn_burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    gootput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
    rag_dataset: Optional[str],
    rag_model: Optional[str],
    top_k: Optional[int]
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    input_requests = generate_requests(input_requests, tokenizer)

    if rag_dataset is not None and rag_model is not None:
        input_requests = generate_rag(input_requests, rag_dataset, rag_model, top_k)

    print("Starting initial single prompt test run...")
    test_input = input_requests[0][0]
    # if backend != "openai-chat" and test_mm_content is not None:
    #     # multi-modal benchmark is only available on OpenAI Chat backend.
    #     raise ValueError(
    #         "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input.api_url = api_url
    test_input.model = model_id
    test_input.logprobs = logprobs
    test_input.best_of = best_of
    test_input.ignore_eos = ignore_eos
    # test_input = RequestFuncInput(
    #     model=model_id,
    #     prompt=test_prompt,
    #     api_url=api_url,
    #     prompt_len=test_prompt_len,
    #     output_len=test_output_len,
    #     logprobs=logprobs,
    #     best_of=best_of,
    #     multi_modal_content=test_mm_content,
    #     ignore_eos=ignore_eos,
    # )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = test_input
        profile_input.api_url = base_url + "/start_profile"
        # profile_input = RequestFuncInput(model=model_id,
        #                                  prompt=test_prompt,
        #                                  api_url=base_url + "/start_profile",
        #                                  prompt_len=test_prompt_len,
        #                                  output_len=test_output_len,
        #                                  logprobs=logprobs,
        #                                  best_of=best_of,
        #                                  multi_modal_content=test_mm_content,
        #                                  ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=sum(len(conv) for conv in input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, tokenizer, request_rate, burstiness, turn_request_rate, turn_burstiness):
        request_func_input = request
        request_func_input.api_url = api_url
        request_func_input.model = model_id
        request_func_input.logprobs = logprobs
        request_func_input.best_of = best_of
        request_func_input.ignore_eos = ignore_eos
        # request_func_input = RequestFuncInput(model=model_id,
        #                                       prompt=prompt,
        #                                       api_url=api_url,
        #                                       prompt_len=prompt_len,
        #                                       output_len=output_len,
        #                                       logprobs=logprobs,
        #                                       best_of=best_of,
        #                                       multi_modal_content=mm_content,
        #                                       ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input.api_url = base_url + "/stop_profile"
        # profile_input = RequestFuncInput(
        #     model=model_id,
        #     prompt=test_prompt,
        #     api_url=base_url + "/stop_profile",
        #     prompt_len=test_prompt_len,
        #     output_len=test_output_len,
        #     logprobs=logprobs,
        #     best_of=best_of,
        # )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        gootput_config_dict=gootput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if gootput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        metrics.request_goodput if gootput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    gootput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        gootput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in gootput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return gootput_config_dict


def parse_goodput(slo_pairs):
    gootput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            gootput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return gootput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    gootput_config_dict = check_goodput_args(args)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            turn_request_rate=args.turn_request_rate,
            turn_burstiness=args.turn_burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            gootput_config_dict=gootput_config_dict,
            max_concurrency=args.max_concurrency,
            rag_dataset=args.rag_dataset,
            rag_model=args.rag_model,
            top_k=args.top_k
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")
    
    parser.add_argument(
        "--turn-request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second for each turn in a conversation. "
        "If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--turn-burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation for each turn in a conversation. "
        "Only take effect when turn_request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument(
        "--rag-dataset",
        type=str,
        default=None,
        help="Path to the RAG dataset.",
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default=None,
        help="RAG model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k value for RAG.",
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    args = parser.parse_args()
    main(args)
