from __future__ import annotations

import json
import os
import openai
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import re
from collections import defaultdict
from dataclasses import dataclass

from finegrained_conf.prompts.answer_prompts import *
from finegrained_conf.datasets.data_utils import load_dataset
from finegrained_conf.llm.token_utils import *
from finegrained_conf.utils.parser import *
from finegrained_conf.evaluation.llm_evaluator import *
from finegrained_conf.evaluation.metrics import *
from finegrained_conf.io.run_metadata import ExperimentRecorder, TestInstance
from finegrained_conf.llm.api_wrapper import APIWrapper, IncompleteResponseError
from finegrained_conf.llm.openai_client import *
from finegrained_conf.experiments.utils import normalize_answer, is_same_ans, entropy_nat

@dataclass
class MethodResult:
    answer: Optional[str]
    answer_prob: Any
    p_hat_ans: Optional[dict[str, float]] = None
    H_nat_ans: Optional[float] = None


def estimate_freq_probs_ans(
    model_name,
    prompt,
    n_samples=10,
    T=0.7,
    top_p=0.95,
    *,
    method_name: str | None = None,
    recorder: ExperimentRecorder | None = None,
    test_instance: TestInstance | None = None,
    api_wrapper: APIWrapper | None = None,
):

    ans_counter = defaultdict(int)
    llm_call = lambda: get_model_response(
            model_name,
            prompt,
            temperature=T,
            top_p=top_p,
            n=n_samples,
            method_name=method_name,
            recorder=recorder,
            test_instance=test_instance,
        )
    if api_wrapper is not None:
        res_list = api_wrapper.call_llm(
            llm_call,
            test_instance,
            method_name or "label_prob",
            "confidence_estimation",
            sample_index=0,
        )
    else:
        res_list = llm_call()
    if not res_list:
        return {}
    for txt in res_list:
        res = parse_output(txt)
        print(res)
        ans = res.answers[0] if res.answers else None
        if ans: ans_counter[normalize_answer(ans)] += 1

    N_ans = sum(ans_counter.values())
    if N_ans == 0:
        return {}

    print(ans_counter)
    p_hat_ans = {a: cnt/N_ans for a, cnt in ans_counter.items()}

    return p_hat_ans


def _handle_label_prob(
    method,
    question,
    templates,
    model_name,
    language,
    debug,
    recorder,
    current_test_instance,
    api_wrapper=None,
):

    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    p_hat_ans = estimate_freq_probs_ans(
        model_name,
        prompt,
        method_name=method,
        recorder=recorder,
        test_instance=current_test_instance,
        api_wrapper=api_wrapper,
    )
    if not p_hat_ans:
        raise IncompleteResponseError("Failed to estimate frequency probabilities: p_hat_ans is empty")
    answer, answer_prob = max(p_hat_ans.items(), key=lambda x: x[1])
    H_nat_ans = entropy_nat(p_hat_ans)
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    return MethodResult(answer, answer_prob, p_hat_ans, H_nat_ans)


def _handle_label_prob_cot_binary(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
    prompt1s = templates["label_prob_cot"].format(THE_QUESTION=question)
    prompt2s = templates["is_true_logprob_binary"]
    res, confidence, response2 = run_two_step_evaluation(
        model_name, prompt1s, prompt2s, logprobs=True, debug=debug, recorder=recorder, test_instance=test_instance
    )
    true_probabilities = extract_binary_probabilities(response2, "1", target_text=("1", "0"))
    if not res.answers:
        raise IncompleteResponseError("No answers found in response")
    answer = res.answers[0]
    answer_prob = None
    for k, v in true_probabilities.items():
        if "回答" in k:
            answer_prob = v["true_probability"]
            break
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer probability from true_probabilities")
    return MethodResult(answer, answer_prob)


def _handle_logprob(
    method, question, templates, get_logprob_response, language, debug
):
    prompt = templates["label_prob"].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response_text, logprobs_info = get_logprob_response(
        prompt,
        temperature=0.0,
        top_logprobs=5,
    )
    if not response_text or not logprobs_info:
        raise IncompleteResponseError("Failed to get response_text or logprobs_info")
    res = calculate_all_triple_confidences(
        logprobs_info["tokens"],
        logprobs_info["logprobs"],
        forcing_match_answer=True,
        language=language,
    )
    answer = res["answer"]["text"]
    answer_prob = res["answer"]["confidence"]
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    return MethodResult(answer, answer_prob)


def _handle_logprob_cot(
    method, question, templates, get_logprob_response, language, debug
):
    prompt = templates["verb_1s_cot"].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response_text, logprobs_info = get_logprob_response(
        prompt,
        temperature=0.0,
        top_logprobs=5,
    )
    if not response_text or not logprobs_info:
        raise IncompleteResponseError("Failed to get response_text or logprobs_info")
    res = calculate_all_triple_confidences(
        logprobs_info["tokens"],
        logprobs_info["logprobs"],
        forcing_match_answer=True,
        language=language,
    )
    if not res["answer"]:
        raise IncompleteResponseError("No answer found in triple confidences")
    answer = res["answer"]["text"]
    answer_prob = res["answer"]["confidence"]
    res2 = parse_output(response_text)
    if res2.answer_conf:
        answer_prob["verb_prob"] = res2.answer_conf[0]
    else:
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    return MethodResult(answer, answer_prob)


def _handle_is_true_prob_or_ling_2s(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
    prompt1s = templates["label_prob"].format(THE_QUESTION=question)
    prompt2s = templates[method]
    res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    if not res.answers:
        raise IncompleteResponseError("No answers found in response")
    answer = res.answers[0]
    answer_prob = confidence.answer_conf[0] if confidence and confidence.answer_conf else None
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer probability from confidence")
    return MethodResult(answer, answer_prob)


# def _handle_is_true_logprob(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
#     prompt1s = templates["verb_2s_top_1"].format(THE_QUESTION=question)
#     prompt2s = templates["is_true_logprob"]
#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True, debug=debug, recorder=recorder, test_instance=test_instance)
#     true_probabilities = extract_true_probabilities(response2)
#     if not res.answers:
#         raise IncompleteResponseError("No answers found in response")
#     answer = res.answers[0]
#     answer_prob = None
#     for k, v in true_probabilities.items():
#         if "回答" in k:
#             answer_prob = v["true_probability"]
#             break
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
#     if answer_prob is None:
#         raise IncompleteResponseError("Failed to extract answer probability from true_probabilities")
#     return MethodResult(answer, answer_prob)


# def _handle_is_true_2s_logprob(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
#     prompt1s = templates["label_prob"].format(THE_QUESTION=question)
#     prompt2s = templates["is_true_logprob"]
#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True, debug=debug, recorder=recorder, test_instance=test_instance)
#     true_probabilities = extract_true_probabilities(response2)
#     if not res.answers:
#         raise IncompleteResponseError("No answers found in response")
#     answer = res.answers[0]
#     answer_prob = None
#     for k, v in true_probabilities.items():
#         if "回答" in k:
#             answer_prob = v["true_probability"]
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
#     if answer_prob is None:
#         raise IncompleteResponseError("Failed to extract answer probability from true_probabilities")
#     return MethodResult(answer, answer_prob)


# def _handle_is_true_cot1s_logprob(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
#     prompt1s = templates["verb_1s_cot"].format(THE_QUESTION=question)
#     prompt2s = templates["is_true_logprob"]
#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True, debug=debug, recorder=recorder, test_instance=test_instance)
#     if not res.answers:
#         raise IncompleteResponseError("No answers found in response")
#     answer = res.answers[0]
#     _answer_prob = res.answer_conf[0] if res.answer_conf else None
#     true_probabilities = extract_true_probabilities(response2)
#     answer_prob = {
#         "is_true_cot1s_logprob_s1_prob": _answer_prob,
#         "is_true_cot1s_logprob_s2_prob": None,
#     }
#     for k, v in true_probabilities.items():
#         if "回答" in k:
#             answer_prob["is_true_cot1s_logprob_s2_prob"] = v["true_probability"]
#     answer_prob["is_true_cot1s_logprob_prob"] = (
#         answer_prob["is_true_cot1s_logprob_s1_prob"]
#         * answer_prob["is_true_cot1s_logprob_s2_prob"]
#     )
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
#     if answer_prob["is_true_cot1s_logprob_s2_prob"] is None:
#         raise IncompleteResponseError("Failed to extract is_true_cot1s_logprob_s2_prob from true_probabilities")
#     return MethodResult(answer, answer_prob)


# def _handle_is_true_cot2s_logprob(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None, recorder=None):
#     prompt1s = templates["verb_2s_cot"].format(THE_QUESTION=question)
#     prompt2s = templates["is_true_logprob"]
#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True, debug=debug, recorder=recorder, test_instance=test_instance)
#     if not res.answers:
#         raise IncompleteResponseError("No answers found in response")
#     answer = res.answers[0]
#     true_probabilities = extract_true_probabilities(response2)
#     answer_prob = None
#     for k, v in true_probabilities.items():
#         if "回答" in k:
#             answer_prob = v["true_probability"]
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
#     if answer_prob is None:
#         raise IncompleteResponseError("Failed to extract answer probability from true_probabilities")
#     return MethodResult(answer, answer_prob)


# def _handle_verb_1s_cot_is_true(method, question, templates, model_name, debug, api_wrapper=None, test_instance=None):
#     prompt = templates["verb_1s_cot_is_true"].format(THE_QUESTION=question)
#     if debug:
#         print(f"prompt:{prompt}")
#     client = openai.OpenAI()
#     llm_call = lambda: client.chat.completions.create(
#         model=model_name,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=512,
#         logprobs=True,
#         top_logprobs=5,
#         top_p=1.0,
#     )
#     response = (
#         api_wrapper.call_llm(
#             llm_call,
#             test_instance,
#             method,
#             "confidence_estimation",
#             sample_index=0,
#         )
#         if api_wrapper is not None
#         else llm_call()
#     )
#     # コンテンツフィルタリングエラーでNoneが返された場合はエラーを投げる
#     if not response:
#         raise IncompleteResponseError("No response from LLM")
#     response_text = response.choices[0].message.content.strip()
#     res = parse_output(response_text)
#     if not res.answers:
#         raise IncompleteResponseError("No answers found in parsed output")
#     answer = res.answers[0]
#     true_probabilities = extract_true_probabilities(response)
#     answer_prob = None
#     for k, v in true_probabilities.items():
#         if "回答" in k:
#             answer_prob = v["true_probability"]
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
#     if answer_prob is None:
#         raise IncompleteResponseError("Failed to extract answer probability from true_probabilities")
#     return MethodResult(answer, answer_prob)


def _handle_verb_1s_top(method, question, templates, get_response, debug):
    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        raise IncompleteResponseError("No response from LLM")
    res = parse_output(response)
    if not res.answers:
        raise IncompleteResponseError("No answers found in parsed output")
    answer = res.answers[0]
    answer_prob = None
    if res.answer_conf:
        answer_prob = res.answer_conf[0]
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")
    return MethodResult(answer, answer_prob)


def _handle_verb_1s_cot(method, question, templates, get_response, debug):
    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        raise IncompleteResponseError("No response from LLM")
    res = parse_output(response)
    if not res.answers:
        raise IncompleteResponseError("No answers found in parsed output")
    answer = res.answers[0]
    answer_prob = None
    if res.answer_conf:
        answer_prob = res.answer_conf[0]
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")
    return MethodResult(answer, answer_prob)


def _handle_verb_2s(method, question, templates, model_name, debug, recorder=None, test_instance=None):
    if "top_1" in method:
        prompt1s = templates["verb_2s_top_1"].format(THE_QUESTION=question)
        prompt2s = templates["verb_2s_top_1_prob"]
        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    elif "top_" in method:
        k = int(method.split("_")[-1])
        prompt1s = templates["verb_2s_top_k"].format(THE_QUESTION=question, k=k)
        if debug:
            print(f"prompt:{prompt1s}")
        prompt2s = templates["verb_2s_top_k_prob"].format(k=k)
        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    elif "cot" in method:
        prompt1s = templates[method].format(THE_QUESTION=question)
        if debug:
            print(f"prompt:{prompt1s}")
        prompt2s = templates["verb_2s_cot_prob"]
        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    else:
        prompt1s = templates[method].format(THE_QUESTION=question)
        if debug:
            print(f"prompt:{prompt1s}")
        prompt2s = templates["verb_2s_top_1_prob"]
        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    if not res.answers:
        raise IncompleteResponseError("No answers found in response")
    answer = res.answers[0]
    answer_prob = confidence.answer_conf[0] if confidence and confidence.answer_conf else None
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer_conf from confidence")
    return MethodResult(answer, answer_prob)


def _handle_ling_1s(method, question, templates, language, get_response, debug):
    if language == "en":
        prompt = templates["ling_1s_human"].format(
            THE_QUESTION=question, EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS
        )
    elif method == "ling_1s_human7":
        prompt = templates["ling_1s_human"].format(
            THE_QUESTION=question, EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_7JP
        )
    else:
        prompt = templates["ling_1s_human"].format(
            THE_QUESTION=question, EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_JP
        )
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        raise IncompleteResponseError("No response from LLM")
    res = parse_output(response)
    if not res.answers:
        raise IncompleteResponseError("No answers found in parsed output")
    answer = res.answers[0]
    answer_prob = None
    if res.answer_conf:
        answer_prob = res.answer_conf[0]
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")
    return MethodResult(answer, answer_prob)


def _handle_ling_2s(method, question, templates, model_name, debug, recorder=None, test_instance=None):
    prompt1s = templates["verb_2s_top_1"].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt1s}")
    if method == "ling_2s_human7":
        prompt2s = templates["ling_2s_human"].format(
            THE_QUESTION=question, EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_7JP
        )
    else:
        prompt2s = templates[method].format(
            THE_QUESTION=question, EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_JP
        )
    res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    if not res.answers:
        raise IncompleteResponseError("No answers found in response")
    answer = res.answers[0]
    answer_prob = confidence.answer_conf[0] if confidence and confidence.answer_conf else None
    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
    if answer_prob is None:
        raise IncompleteResponseError("Failed to extract answer_conf from confidence")
    return MethodResult(answer, answer_prob)

def _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder=None, test_instance=None, budget_tokens_reasoning = 100):
    """
    Budgeted 2-step baseline for reasoning models (DeepSeek-R1, etc.)

    Turn0: Stream call with budget-based truncation (default: 100 tokens)
    Turn1: Final output with Turn0 reasoning as context
    """
    from finegrained_conf.llm.api_wrapper import run_budgeted_reasoning_2turn

    answer, answer_prob = None, None


    prompt_start = templates["budgeted_2step_start"].format(THE_QUESTION=question)
    final_prompt = templates["budgeted_2step_final"]

    max_tokens_reasoning_server = 4096
    max_tokens_final = 1024
    temperature_reasoning = 0.0
    temperature_final = 0.0

    if debug:
        print(f"Budgeted 2-step baseline method: {method}")
        print(f"prompt_start: {prompt_start}")
        print(f"final_prompt: {final_prompt}")
        print(f"budget_tokens_reasoning: {budget_tokens_reasoning}")

    # Call run_budgeted_reasoning_2turn.
    try:
        result = run_budgeted_reasoning_2turn(
            model_name=model_name,
            prompt_start=prompt_start,
            final_prompt=final_prompt,
            budget_tokens_reasoning=budget_tokens_reasoning,
            max_tokens_reasoning_server=max_tokens_reasoning_server,
            max_tokens_final=max_tokens_final,
            temperature_reasoning=temperature_reasoning,
            temperature_final=temperature_final,
            # stop_reasoning=["[OUTPUT]", "[/OUTPUT]"],
            stop_reasoning=["</think>"],
            stop_final=["[/OUTPUT]"],
            recorder=recorder,
            test_instance=test_instance,
            method_name=method,
            log_reasoning=True,
        )
    except Exception as e:
        if debug:
            print(f"Error in run_budgeted_reasoning_2turn: {e}")
        raise IncompleteResponseError(f"Budgeted 2-step baseline failed: {e}")

    final_response = result["final_response_text"]

    if not final_response:
        if debug:
            print('final_response is None')
        raise IncompleteResponseError("final_response is None")

    # The parser extracts and parses only the [OUTPUT] block.
    res = parse_output(final_response)

    if not res.answers:
        if debug:
            print('res.answers is None')
        raise IncompleteResponseError("No answers found in parsed output")

    answer = res.answers[0]
    if res.answer_conf:
        answer_prob = res.answer_conf[0]

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")
        print(f"Turn0 estimated tokens: {result['turn0_est_tokens']}")

    if answer_prob is None:
        if debug:
            print('answer_prob is None')
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")

    return MethodResult(answer, answer_prob)

def _handle_budgeted_1step(method, question, templates, model_name, debug, recorder=None, test_instance=None):
    """
    Budgeted 2-step baseline for reasoning models (DeepSeek-R1, etc.)

    Turn0: Stream call with budget-based truncation (default: 100 tokens)
    Turn1: Final output with Turn0 reasoning as context
    """
    from finegrained_conf.llm.api_wrapper import run_budgeted_reasoning_1turn

    answer, answer_prob = None, None

    # Get the prompt template
    final_prompt = templates["budgeted_1step_final"].format(THE_QUESTION=question)

    # Default parameters
    # budget_tokens_reasoning = 100  # Reasoning token budget (default)
    max_tokens_final = 1024  # Max tokens for the final turn
    temperature_final = 0.0  # Temperature for the final turn

    if debug:
        print(f"Budgeted 1-step method: {method}")
        print(f"final_prompt: {final_prompt}")

    # Call run_budgeted_reasoning_2turn.
    try:
        result = run_budgeted_reasoning_1turn(
            model_name=model_name,
            final_prompt=final_prompt,
            max_tokens_final=max_tokens_final,
            temperature_final=temperature_final,
            stop_final=["[/OUTPUT]"],
            recorder=recorder,
            test_instance=test_instance,
            method_name=method,
            log_reasoning=True,
        )
    except Exception as e:
        if debug:
            print(f"Error in run_budgeted_reasoning_1turn: {e}")
        raise IncompleteResponseError(f"Budgeted 1-step failed: {e}")

    final_response = result["final_response_text"]

    if not final_response:
        if debug:
            print('final_response is None')
        raise IncompleteResponseError("final_response is None")

    # The parser extracts and parses only the [OUTPUT] block.
    res = parse_output(final_response)

    if not res.answers:
        if debug:
            print('res.answers is None')
        raise IncompleteResponseError("No answers found in parsed output")

    answer = res.answers[0]
    if res.answer_conf:
        answer_prob = res.answer_conf[0]

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}")

    if answer_prob is None:
        if debug:
            print('answer_prob is None')
        raise IncompleteResponseError("Failed to extract answer_conf from parsed output")

    return MethodResult(answer, answer_prob)

def execute_method(
    method,
    question,
    templates,
    language,
    model_name,
    debug,
    get_response,
    get_logprob_response,
    recorder,
    current_test_instance,
    api_wrapper=None,
):
    handler_map = {
        "label_prob": _handle_label_prob,
        "label_prob_cot": _handle_label_prob,
        "label_prob_cot_binary": _handle_label_prob_cot_binary,
        "logprob": _handle_logprob,
        "logprob_cot": _handle_logprob_cot,
        "is_true_prob": _handle_is_true_prob_or_ling_2s,
        "ling_2s_human": _handle_is_true_prob_or_ling_2s,
        # "is_true_logprob": _handle_is_true_logprob,
        # "is_true_2s_logprob": _handle_is_true_2s_logprob,
        # "is_true_cot1s_logprob": _handle_is_true_cot1s_logprob,
        # "is_true_cot2s_logprob": _handle_is_true_cot2s_logprob,
        # "verb_1s_cot_is_true": _handle_verb_1s_cot_is_true,
    }

    if method in handler_map:
        handler = handler_map[method]
        if handler is _handle_label_prob:
            return handler(
                method,
                question,
                templates,
                model_name,
                language,
                debug,
                recorder,
                current_test_instance,
                api_wrapper,
            )
        if handler in {
            _handle_label_prob_cot_binary,
            _handle_is_true_prob_or_ling_2s,
            # _handle_is_true_logprob,
            # _handle_is_true_2s_logprob,
            # _handle_is_true_cot1s_logprob,
            # _handle_is_true_cot2s_logprob,
            # _handle_verb_1s_cot_is_true,
        }:
            return handler(method, question, templates, model_name, debug, api_wrapper, current_test_instance, recorder)
        return handler(method, question, templates, get_logprob_response, language, debug)

    if method == "budgeted_2step_baseline":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance)
    if method == "budgeted_2step_100":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance, budget_tokens_reasoning=100)
    if method == "budgeted_2step_200":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance, budget_tokens_reasoning=200)
    if method == "budgeted_2step_500":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance, budget_tokens_reasoning=500)
    if method == "budgeted_2step_750":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance, budget_tokens_reasoning=750)
    if method == "budgeted_2step_1000":
        return _handle_budgeted_2step_baseline(method, question, templates, model_name, debug, recorder, current_test_instance, budget_tokens_reasoning=1000)
    if method == "budgeted_1step_baseline":
        return _handle_budgeted_1step(method, question, templates, model_name, debug, recorder, current_test_instance)
    if method == "verb_1s_top_1":
        return _handle_verb_1s_top(method, question, templates, get_response, debug)
    if method.startswith("verb_1s_top_"):
        return _handle_verb_1s_top(method, question, templates, get_response, debug)
    if method == "verb_1s_cot":
        return _handle_verb_1s_cot(method, question, templates, get_response, debug)
    if method.startswith("verb_2s_"):
        return _handle_verb_2s(method, question, templates, model_name, debug, recorder, current_test_instance)
    if method in ["ling_1s_human", "ling_1s_human7"]:
        return _handle_ling_1s(method, question, templates, language, get_response, debug)
    if method in ["ling_2s_human", "ling_2s_human7"]:
        return _handle_ling_2s(method, question, templates, model_name, debug, recorder, current_test_instance)

    raise IncompleteResponseError(f"Unknown method: {method}")

def run_experiment(
    model_name,
    dataset_name,
    split,
    methods,
    num_samples=100,
    language="en",
    test_suffix="",
    overwrite=True,
    debug=False,
    recorder=None,
    test_instances=None,
):
    """
    Run the experiment and collect the results.

    Args:
        model_name: Name of the model to use.
        dataset_name: Name of the dataset to use.
        methods: List of methods to run.
        num_samples: Number of samples to use.
        language: Language to use ("en" or "ja").

    Returns:
        A dictionary containing the results for each method.
    """

    dataset = load_dataset(dataset_name, split=split, max_samples=num_samples)
    api_wrapper = APIWrapper(recorder=recorder)
    if debug:
        print(dataset_name, split, num_samples)
        for i in range(min(3, len(dataset))):
            sample = dataset.get_sample(i)
            # print(sample)
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['answer']}")
            if 'derivations' in sample:
                print(f"  Derivations: {sample['derivations']}")
            print()

    if language == "ja":
        templates = PROMPT_TEMPLATES_JP
        linguistic_to_prob = LING_JA
        linguistic_expressions = LINGUISTIC_EXPRESSIONS_JP
        evaluation_prompt_template = EVALUATION_PROMPT_JP
    else:
        templates = PROMPT_TEMPLATES_EN
        linguistic_to_prob = LING_EN
        linguistic_expressions = LINGUISTIC_EXPRESSIONS
        evaluation_prompt_template = EVALUATION_PROMPT_EN

    results = {}
    tmp_result_file = f"tmp_results/results_answer_{model_name}_{dataset_name}_{split}_{test_suffix}_{num_samples}.json"

    for method in methods:
        print(f"Running method: {method} on {model_name} with {dataset_name} in {language}")
        tmp_method_result_file = f"tmp_results/results_{model_name}_{dataset_name}_{split}_{method}{test_suffix}_{num_samples}.json"

        if not overwrite and os.path.isfile(tmp_method_result_file):
            with open(tmp_method_result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if method in ["logprob", "is_true_logprob", "is_true_cot1s_logprob", "logprob_cot"]:
                    tmp_result = {}
                    for k, v in data.items():
                        if k.endswith('prob'):
                            if method in k:
                                tmp_result[k] = v
                            else:
                                tmp_result[method+"_"+k] = v
                else:
                    tmp_result = {method:data[method]}

        else:
            answers = []
            answers_probs = []
            correctness = []
            p_hat_ans_list, H_nat_ans_list = [], []

            tmp_qa_result_file = f"tmp_results/results_qa_{model_name}_{dataset_name}_{split}_{method}{test_suffix}_{num_samples}.json"
            if not overwrite and os.path.isfile(tmp_qa_result_file):
                print(f"load tmp_qa_result_file : {tmp_qa_result_file}")
                with open(tmp_qa_result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    answers = data["answers"]
                    answers_probs = data["answer_confidences"]
                    correctness = data["answer_correctness"]

            for i in tqdm(range(len(dataset))):
                if len(answers) > i:
                    continue
                sample = dataset.get_sample(i)
                question = sample["question"]
                gold_answer = sample["answer"]
                answer, answer_prob = None, None
                current_test_instance = None
                if test_instances is not None and i < len(test_instances):
                    current_test_instance = test_instances[i]

                if debug:
                    print(f"question:{question}, gold_answer:{gold_answer}")

                def get_response(prompt, **kwargs):
                    llm_call = lambda: get_model_response(
                        model_name,
                        prompt,
                        method_name=method,
                        recorder=recorder,
                        test_instance=current_test_instance,
                        **kwargs,
                    )
                    response = api_wrapper.call_llm(
                        llm_call,
                        current_test_instance,
                        method,
                        "answer_generation",
                        sample_index=0,
                    )
                    if debug:
                        print(f"response:{response}")

                    return response

                def get_logprob_response(prompt, **kwargs):
                    result = api_wrapper.call_llm(
                        lambda: get_response_with_logprobs(
                            model_name,
                            prompt,
                            method_name=method,
                            recorder=recorder,
                            test_instance=current_test_instance,
                            **kwargs,
                        ),
                        current_test_instance,
                        method,
                        "confidence_estimation",
                        sample_index=0,
                    )
                    # If a content filtering error occurs, return (None, None).
                    if result is None:
                        return (None, None)
                    return result

                answer, answer_prob = None, None
                p_hat_ans, H_nat_ans = None, None

                try:
                    method_result = execute_method(
                        method,
                        question,
                        templates,
                        language,
                        model_name,
                        debug,
                        get_response,
                        get_logprob_response,
                        recorder,
                        current_test_instance,
                        api_wrapper,
                    )
                    answer = method_result.answer
                    answer_prob = method_result.answer_prob
                    p_hat_ans = method_result.p_hat_ans
                    H_nat_ans = method_result.H_nat_ans
                except IncompleteResponseError as e:
                    # If no response was obtained due to content filtering or other reasons.
                    print(f'WARNING: IncompleteResponseError - {e}, skipping question:', question, gold_answer)
                    answer = None
                    answer_prob = None
                    p_hat_ans = None
                    H_nat_ans = None

                if answer:
                    # Check whether the answer is correct.
                    is_correct = check_answer_correctness(
                        answer, gold_answer, question, model="gpt-4.1-2025-04-14", language=language, debug=debug
                    )
                else:
                    is_correct = None

                answers.append(answer)
                answers_probs.append(answer_prob)
                correctness.append(1 if is_correct else 0)
                if method == "label_prob":
                    p_hat_ans_list.append(p_hat_ans)
                    H_nat_ans_list.append(H_nat_ans)

                tmp_result = {
                    "answers": answers,
                    "answer_confidences": answers_probs,
                    "answer_correctness": correctness,
                }

                if method == "label_prob":
                    tmp_result["p_hat_ans"] = p_hat_ans_list
                    tmp_result["H_nat_ans"] = H_nat_ans_list

                # Save the results.
                with open(tmp_qa_result_file, "w") as f:
                    json.dump(tmp_result, f, ensure_ascii=False)

            def format_answer(answers, answers_probs, answer_correctness):
                print(answers, answers_probs, answer_correctness)
                print('answers_probs', answers_probs, type(answers_probs))
                print('answer_correctness', answer_correctness, type(answer_correctness))

                res = {
                    "answers": answers,
                    "answer_confidences": answers_probs,
                    "answer_correctness": answer_correctness,
                    "answer_metrics": None,
                }
                return res

            if method in ["logprob", "is_true_cot1s_logprob", "logprob_cot"]:
                tmp_result = {}
                answers_prob_dict = defaultdict(list)
                for a_dic in answers_probs:
                    if not a_dic:
                        for k in answers_probs[0].keys():
                            answers_prob_dict[k].append(None)
                    else:
                        for k, v in a_dic.items():
                            answers_prob_dict[k].append(v)
                            if k == 'tokens':
                                answers_prob_dict['answers'].append(''.join(v))
                for k, v in answers_prob_dict.items():
                    if k.endswith('prob'):
                        if method in k:
                            tmp_method_name = k
                        else:
                            tmp_method_name = method+"_"+k
                        res = format_answer(answers, v, correctness)
                        if method == "label_prob":
                            res["p_hat_ans"] = p_hat_ans_list
                            res["H_nat_ans"] = H_nat_ans_list
                        tmp_result[tmp_method_name] = res
            else:
                res = format_answer(answers, answers_probs, correctness)
                if method == "label_prob":
                    res["p_hat_ans"] = p_hat_ans_list
                    res["H_nat_ans"] = H_nat_ans_list
                tmp_result = {method:res}

            # Save the results.
            with open(tmp_method_result_file, "w") as f:
                json.dump(tmp_result, f, ensure_ascii=False)

        for k, v in tmp_result.items():
            v["answer_metrics"] = compute_metrics(v["answer_confidences"], v["answer_correctness"], model_name=k)
            results[k] = v

        with open(tmp_result_file, "w") as f:
            json.dump(results, f, ensure_ascii=False)

    return results