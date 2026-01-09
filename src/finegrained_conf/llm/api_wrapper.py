#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM API呼び出しのラッパー関数

すべてのLLM呼び出しを ExperimentRecorder でログ記録するためのラッパー。
"""

from typing import Any, Dict, List, Optional, Union
import time
from functools import wraps

from finegrained_conf.io.run_metadata import ExperimentRecorder, TestInstance


class IncompleteResponseError(Exception):
    """
    Exception raised when an LLM response is incomplete or invalid.

    This exception is used in the following cases:
    - Failed to parse the response
    - Required information is missing from the response
    - Could not extract the answer or confidence

    The APIWrapper catches this exception and automatically retries.
    """
    pass


class ContentFilterError(Exception):
    """
    Exception raised when a response is rejected due to content filtering.

    This exception is used in the following cases:
    - The API returns None content
    - The response is blocked due to content filtering

    The APIWrapper catches this exception and skips the request without retrying.
    """
    pass


def with_api_logging(
    recorder: Optional[ExperimentRecorder],
    test_instance: Optional[TestInstance],
    method: str,
    call_role: str,  # "answer_generation" | "confidence_estimation" | "evaluation" | "retry"
    sample_index: int = 0,
):
    """
    Decorator for logging LLM API calls
    
    Usage:
        @with_api_logging(recorder, test_instance, method, "answer_generation", sample_idx)
        def get_answer(model_name, prompt, **kwargs):
            return get_model_response(model_name, prompt, **kwargs)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if recorder is None or test_instance is None:
                return func(*args, **kwargs)
            
            attempt = kwargs.pop('_attempt', 0)
            start_time = time.time()
            
            try:
                response = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                payload = {
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:500] for k, v in kwargs.items()},
                    "response_preview": str(response)[:1000],
                    "elapsed_time": elapsed_time,
                }
                
                recorder.log_api_call(
                    method=method,
                    test_instance=test_instance,
                    call_role=call_role,
                    sample_index=sample_index,
                    attempt=attempt,
                    status="success",
                    payload=payload,
                )
                
                return response
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                
                payload = {
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:500] for k, v in kwargs.items()},
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "elapsed_time": elapsed_time,
                }
                
                recorder.log_api_call(
                    method=method,
                    test_instance=test_instance,
                    call_role=call_role,
                    sample_index=sample_index,
                    attempt=attempt,
                    status="error",
                    payload=payload,
                )
                
                raise
        
        return wrapper
    return decorator


class APIWrapper:
    """
    Wrapper class for LLM API calls.

    Logs all LLM calls in a unified manner.
    """
    
    def __init__(
        self,
        recorder: Optional[ExperimentRecorder] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.recorder = recorder
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def call_llm(
        self,
        llm_func,
        test_instance: Optional[TestInstance],
        method: str,
        call_role: str,
        sample_index: int = 0,
        *args,
        **kwargs,
    ) -> Any:
        """
        Call the LLM function and record logs.

        Args:
            llm_func: The actual LLM call function.
            test_instance: The test instance.
            method: Experiment method name.
            call_role: Role of this call.
            sample_index: Sample index.
            *args, **kwargs: Arguments passed to llm_func.

        Returns:
            The result of the LLM call.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if self.recorder and test_instance:
                    start_time = time.time()
                    response = llm_func(*args, **kwargs)
                    elapsed_time = time.time() - start_time
                    
                    self.recorder.log_api_call(
                        method=method,
                        test_instance=test_instance,
                        call_role=call_role,
                        sample_index=sample_index,
                        attempt=attempt,
                        status="success",
                        payload={
                            "response_preview": str(response)[:1000],
                            "elapsed_time": elapsed_time,
                            "args_preview": str(args)[:500],
                            "kwargs_keys": list(kwargs.keys()),
                        },
                    )
                else:
                    response = llm_func(*args, **kwargs)
                
                return response
                
            except Exception as e:
                last_exception = e

                is_content_filter_error = (
                    isinstance(e, ValueError) and
                    "API returned None content" in str(e)
                )

                if self.recorder and test_instance:
                    self.recorder.log_api_call(
                        method=method,
                        test_instance=test_instance,
                        call_role=call_role,
                        sample_index=sample_index,
                        attempt=attempt,
                        status="skipped" if is_content_filter_error else "error",
                        payload={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "args_preview": str(args)[:500],
                            "kwargs_keys": list(kwargs.keys()),
                            "is_content_filter_error": is_content_filter_error,
                        },
                    )

                if is_content_filter_error:
                    print(f"Warning: Content filtering detected. Skipping this test case.")
                    return None

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        if last_exception:
            raise last_exception



def example_usage():
    from finegrained_conf.io.run_metadata import ExperimentRecorder, TestInstance
    from finegrained_conf.llm.openai_client import get_model_response
    
    recorder = ExperimentRecorder(
        run_id="jemhop_qa-dev-gpt-4.1-mini-label_prob-20241124120000",
        model="gpt-4.1-mini-2025-04-14",
        dataset="jemhop_qa",
        split="dev",
    )
    
    test_instance = TestInstance(
        dataset="jemhop_qa",
        split="dev",
        question_index=0,
        question_id="q001",
        dataset_native_id="jemhop_001",
    )
    
    api_wrapper = APIWrapper(recorder=recorder, max_retries=3)
    
    
    prompt = "質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？"
    
    response = api_wrapper.call_llm(
        llm_func=get_model_response,
        test_instance=test_instance,
        method="label_prob",
        call_role="answer_generation",
        sample_index=0,
        model_name="gpt-4.1-mini-2025-04-14",
        prompt=prompt,
        temperature=0.7,
        n=10,
    )
    
    
    @with_api_logging(
        recorder=recorder,
        test_instance=test_instance,
        method="label_prob",
        call_role="confidence_estimation",
        sample_index=0,
    )
    def get_confidence(model_name, prompt, **kwargs):
        return get_model_response(model_name, prompt, **kwargs)
    
    confidence_response = get_confidence(
        model_name="gpt-4.1-mini-2025-04-14",
        prompt="以下の回答の確信度を0.0〜1.0で評価してください：...",
        temperature=0.0,
    )


if __name__ == "__main__":
    print("API wrapper module loaded successfully")
    print("See example_usage() for usage patterns")
