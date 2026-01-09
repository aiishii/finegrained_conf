import numpy as np

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import argparse
import re
import logging
import time
from collections import defaultdict

from finegrained_conf.prompts.answer_prompts import *
from finegrained_conf.prompts.evidence_prompts import *
from finegrained_conf.datasets.data_utils import load_dataset
from finegrained_conf.llm.token_utils import *
from finegrained_conf.llm.openai_client import *
from finegrained_conf.utils.parser import *
from finegrained_conf.evaluation.llm_evaluator import *
from finegrained_conf.evaluation.metrics import *
from finegrained_conf.io.run_metadata import (
    ExperimentRecorder,
    TestInstance,
    build_run_id,
)
from finegrained_conf.llm.api_wrapper import APIWrapper, IncompleteResponseError
from finegrained_conf.experiments.utils import normalize_answer, entropy_nat

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def canon_triple(t):
    # print('canon_triple1', t, type(t))
    if not t:
        return None, None, None
    if type(t) is str:
        t = re.sub(r'\s*[\(（](.+?)[\)）]\s*', r'\1', t)
        # print('canon_triple2', t, type(t))
        t = re.split(r'[,，]', t)
    # print('canon_triple3', t, type(t))
    if len(t) != 3 or None in t:
        return None, None, None
    s,r,o = t[0], t[1], t[2]
    return normalize_answer(s), normalize_answer(r), normalize_answer(o)

def estimate_freq_probs(model_name, prompt, n_samples=10, T=0.7, top_p=0.95, api_wrapper=None, test_instance=None, method_name=None, recorder=None):
    ans_counter = defaultdict(int)        # Answer -> count
    evi_sets = []                         # List of evidence sets (recorded regardless of the answer)
    tri_counter = defaultdict(int)        # per-triple counts

    res_list = []
    for sample_index in range(n_samples):
        llm_call = lambda: get_model_response(
            model_name,
            prompt,
            temperature=T,
            top_p=top_p,
            n=1,
            method_name=method_name or "triple_label_prob",
            recorder=recorder,
            test_instance=test_instance,
        )
        try:
            response = (
                api_wrapper.call_llm(
                    llm_call,
                    test_instance,
                    "triple_label_prob",
                    "confidence_estimation",
                    sample_index=sample_index,
                )
                if api_wrapper is not None
                else llm_call()
            )
        except ValueError as e:
            if "API returned None content" in str(e):
                print(f"Warning: Content filtering detected. Skipping this sample.")
                continue
            else:
                raise
        if response is None:
            continue
        response_text = response[0] if isinstance(response, list) else response
        res_list.append(response_text)
    if not res_list:
        return {}, {}, {}
    for txt in res_list:
        res = parse_output(txt)
        print(res)
        ans = res.answers[0] if res.answers else None
        tris = res.triple_sets[1].triples if res.triple_sets else []
        if not None in tris:
            evi_sets.append(tris)
        if ans: ans_counter[normalize_answer(ans)] += 1
        for t in tris:
            ct = canon_triple(t)
            if not None in ct:
                tri_counter[ct] += 1
    
    N_ans, N_ev = sum(ans_counter.values()), len(evi_sets)
    if N_ans == 0 or N_ev == 0:
        return {}, {}, {}

    print(ans_counter)
    p_hat_ans = {a: cnt/N_ans for a, cnt in ans_counter.items()}

    set_counter = defaultdict(int)
    for tris in evi_sets:
        cts = [canon_triple(t) for t in tris]
        cts = [ct for ct in cts if None not in ct]

        new_key = tuple(sorted(cts))
        set_counter[new_key] += 1

    print(set_counter)
    p_hat_evset = {k: cnt/n_samples for k,cnt in set_counter.items()}
    # total_tri = sum(tri_counter.values())
    p_hat_trip  = {t: cnt / n_samples for t, cnt in tri_counter.items()}

    return p_hat_ans, p_hat_evset, p_hat_trip


def triple_to_str(t: tuple[str, ...]) -> str:
    return '(' + ', '.join(t) + ')'

def tripleset_to_str(ts: tuple[tuple[str,...], ...]) -> str:
    ts_sorted = sorted(ts)
    return ';'.join(triple_to_str(tr) for tr in ts_sorted)

@dataclass
class EvidenceMethodResult:
    answer: Optional[str]
    triples: List[Any]
    answer_prob: Any
    triple_probs: List[Any]
    logprobs_info: Optional[dict] = None
    p_hat_ans: Optional[dict] = None
    p_hat_evset: Optional[dict] = None
    p_hat_trip: Optional[dict] = None
    set_prob: Optional[Any] = None
    H_nat_ans: Optional[float] = None
    H_nat_evset: Optional[float] = None
    H_nat_trip: Optional[float] = None


def _run_triple_label_prob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None, recorder=None, test_instance=None):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates['triple_label_prob'].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    p_hat_ans, p_hat_evset, p_hat_trip = estimate_freq_probs(model_name, prompt, n_samples=10, method_name=method, recorder=recorder, test_instance=test_instance)

    if not p_hat_ans or not p_hat_evset:
        return None
    answer, answer_prob = max(p_hat_ans.items(), key=lambda x: x[1])

    triples_tup, set_prob = max(p_hat_evset.items(), key=lambda x: x[1])

    for t in triples_tup:
        ct = canon_triple(t)
        if ct in p_hat_trip:
            triple_probs.append(p_hat_trip[ct])
            triples.append(f"({', '.join(ct)})")

    H_nat_ans  = entropy_nat(p_hat_ans)
    H_nat_evset= entropy_nat(p_hat_evset)
    H_nat_trip = entropy_nat(p_hat_trip)

    p_hat_evset = { tripleset_to_str(k): v for k, v in p_hat_evset.items() }
    p_hat_trip  = { triple_to_str(k):  v for k, v in p_hat_trip.items() }

    if debug:
        print(f"triple_label_prob:answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
        print(f"triple_label_prob:H_nat_ans:{H_nat_ans}, H_nat_evset:{H_nat_evset}, H_nat_trip:{H_nat_trip}")

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_logprob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None, api_wrapper=None, test_instance=None):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates['triple_label_prob'].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    if get_logprob_response is not None:
        response_text, logprobs_info = get_logprob_response(
            prompt,
            temperature=0.0,
            top_logprobs=5,
        )
    else:
        # fallback for backward compatibility
        response_text, logprobs_info = (
            api_wrapper.call_llm(
                lambda: get_response_with_logprobs(
                    model_name,
                    prompt,
                    temperature=0.0,
                    top_logprobs=5,
                ),
                test_instance,
                method,
                "confidence_estimation",
                sample_index=0,
            )
            if api_wrapper is not None
            else get_response_with_logprobs(
                model_name,
                prompt,
                temperature=0.0,
                top_logprobs=5,
            )
        )
    if not response_text or not logprobs_info:
        answer = '回答不可'
        answer_prob = None
        triples = []
        triple_probs = []
        return EvidenceMethodResult(
            answer, triples, answer_prob, triple_probs, logprobs_info,
            p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
        )
    res = calculate_all_triple_confidences(logprobs_info["tokens"], logprobs_info["logprobs"], language=language)
    if debug:
        print(res)
    answer = res['answer']['text']
    answer_prob = res['answer']['confidence']
    triples = [None] * len(res['triples'])
    triple_probs = [None] * len(res['triples'])
    for d in res['triples']:
        if debug:
            print(len(triples), d['num'])
        triples[d['num']-1] = d['text']
        triple_probs[d['num']-1] = d['confidence']
    if debug:
        print(f"triple_logprob: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_logprob_cot(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None, api_wrapper=None, test_instance=None):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates['triple_verb_1s_cot'].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    if get_logprob_response is not None:
        response_text, logprobs_info = get_logprob_response(
            prompt,
            temperature=0.0,
            top_logprobs=5,
        )
    else:
        # fallback for backward compatibility
        response_text, logprobs_info = (
            api_wrapper.call_llm(
                lambda: get_response_with_logprobs(
                    model_name,
                    prompt,
                    temperature=0.0,
                    top_logprobs=5,
                ),
                test_instance,
                method,
                "confidence_estimation",
                sample_index=0,
            )
            if api_wrapper is not None
            else get_response_with_logprobs(
                model_name,
                prompt,
                temperature=0.0,
                top_logprobs=5,
            )
        )
    if not response_text or not logprobs_info:
        answer = '回答不可'
        answer_prob = None
        triples = []
        triple_probs = []
        return EvidenceMethodResult(
            answer, triples, answer_prob, triple_probs, logprobs_info,
            p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
        )
    res = calculate_all_triple_confidences(logprobs_info["tokens"], logprobs_info["logprobs"], language=language)
    if debug:
        print(res)
    if not res['answer']:
        return None
    answer = res['answer']['text']
    answer_prob = res['answer']['confidence']
    triples = [None] * len(res['triples'])
    triple_probs = [None] * len(res['triples'])
    for d in res['triples']:
        if debug:
            print(len(triples), d['num'])
        triples[d['num']-1] = d['text']
        triple_probs[d['num']-1] = d['confidence']

    res2 = parse_output(response_text)
    if debug:
        print('response_text', response_text)
        print('res2', res2)
    if res2.answer_conf:
        answer_prob_verb = res2.answer_conf[0]
    else:
        answer_prob_verb = None
    if res2.triple_sets:
        triple_probs_verb = res2.triple_sets[1].triple_conf
    else:
        triple_probs_verb = [None] * len(triples)
    answer_prob['verb_prob'] = answer_prob_verb
    for t_idx in range(len(triple_probs)):
        triple_probs[t_idx]['verb_prob'] = triple_probs_verb[t_idx]
    if debug:
        print(f"triple_logprob_cot: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
    if answer_prob is None:
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


# def _run_triple_is_true_logprob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt = templates['triple_is_true_logprob'].format(THE_QUESTION=question)
#     if debug:
#         print(f"prompt:{prompt}")
#     if get_logprob_response is not None:
#         response_text, raw_response = get_logprob_response(
#             prompt,
#             temperature=0.0,
#             top_logprobs=5,
#             return_raw_response=True
#         )
#     else:
#         # fallback for backward compatibility
#         response_text, raw_response = get_response_with_logprobs(
#             model_name,
#             prompt,
#             temperature=0.0,
#             top_logprobs=5,
#             return_raw_response=True
#         )
#     if not raw_response:
#         return None
#     res = parse_output(response_text)
#     true_probabilities = extract_confidence_probabilities(raw_response)
#     if not res.answers:
#         return None
#     else:
#         answer = res.answers[0]
#         triples = res.triple_sets[1].triples
#         if debug:
#             print('true_probabilities', true_probabilities)
#         triple_probs = [None] * len(triples)
#         for line in true_probabilities:
#             if '回答' in line['line_text']:
#                 answer_prob = line['true_probability']
#             elif 'トリプル' in line['line_text']:
#                 match = re.search(r'トリプル(\d+)(確信度)?', line['line_text'])
#                 if match:
#                     triple_num = int(match.group(1))
#                     triple_probs[triple_num-1] = line['true_probability']
#         if debug:
#             print('answer_prob', answer_prob)
#             print('triple_probs', triple_probs)
#         if debug:
#             print(f"triple_is_true_logprob: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


# def _run_triple_is_true_cot1s_logprob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None, recorder=None, test_instance=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt1s = templates["triple_verb_1s_cot"].format(THE_QUESTION=question)
#     prompt2s = templates["triple_is_true_logprob"]
#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True,debug=debug, recorder=recorder, test_instance=test_instance)

#     if not res.answers:
#         return None
#     else:
#         answer = res.answers[0]
#         triples = res.triple_sets[1].triples
#         _answer_prob = res.answer_conf[0]
#         _triple_probs = res.triple_sets[1].triple_conf
#         true_probabilities = extract_true_probabilities(response2)

#         answer_prob = {'triple_is_true_cot1s_logprob_s1_prob': _answer_prob, 'triple_is_true_cot1s_logprob_s2_prob': None}

#         triple_probs = [None] * len(_triple_probs)
#         if debug:
#             print('true_probabilities', true_probabilities)

#         for k, v in true_probabilities.items():
#             if '回答' in k:
#                 answer_prob['triple_is_true_cot1s_logprob_s2_prob'] = v['true_probability']
#             elif 'トリプル' in k:
#                 match = re.search(r'トリプル(\d+)確信度', k)
#                 if match:
#                     triple_num = int(match.group(1))
#                     triple_probs[triple_num-1] = {}
#                     triple_probs[triple_num-1]['triple_is_true_cot1s_logprob_s1_prob'] = _triple_probs[triple_num-1]
#                     triple_probs[triple_num-1]['triple_is_true_cot1s_logprob_s2_prob'] = v['true_probability']
#                     triple_probs[triple_num-1]['triple_is_true_cot1s_logprob_prob'] = _triple_probs[triple_num-1] * v['true_probability']

#         if debug:
#             print('answer_prob', answer_prob)
#             print('triple_probs', triple_probs)
#         answer_prob['triple_is_true_cot1s_logprob_prob'] = answer_prob['triple_is_true_cot1s_logprob_s1_prob'] * answer_prob['triple_is_true_cot1s_logprob_s2_prob']
#         if debug:
#             print(f"triple_is_true_cot1s_logprob: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


# def _run_triple_is_true_cot2s_logprob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None, recorder=None, test_instance=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt1s = templates["triple_verb_2s_cot"].format(THE_QUESTION=question)
#     prompt2s = templates["triple_is_true_logprob"]

#     res, confidence, response2 = run_two_step_evaluation(model_name, prompt1s, prompt2s, logprobs=True,debug=debug, recorder=recorder, test_instance=test_instance)

#     if not res.answers:
#         return None
#     else:
#         answer = res.answers[0]
#         triples = res.triple_sets[1].triples
#         true_probabilities = extract_true_probabilities(response2)
#         if debug:
#             print('true_probabilities', true_probabilities)
#         triple_probs = [None] * len(triples)
#         for k, v in true_probabilities.items():
#             if '回答' in k:
#                 answer_prob = v['true_probability']
#             elif 'トリプル' in k:
#                 match = re.search(r'トリプル(\d+)確信度', k)
#                 if match:
#                     triple_num = int(match.group(1))
#                     triple_probs[triple_num-1] = v['true_probability']
#         if debug:
#             print(f"triple_is_true_cot2s_logprob: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )



# def _run_triple_verb_1s_cot_is_true(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt = templates["triple_verb_1s_cot_is_true"].format(THE_QUESTION=question)
#     if debug:
#         print(f"prompt:{prompt}")
#     client = init_client()
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0,
#         max_tokens=512,
#         logprobs=True,
#         top_logprobs=5,
#         top_p=1.0
#     )
#     response_text = response.choices[0].message.content.strip()

#     if not response:
#         return None
#     res = parse_output(response_text)

#     if not res.answers:
#         return None
#     else:
#         answer = res.answers[0]
#         triples = res.triple_sets[1].triples
#         true_probabilities = extract_true_probabilities(response)
#         if debug:
#             print('true_probabilities', true_probabilities)
#         triple_probs = [None] * len(triples)
#         for k, v in true_probabilities.items():
#             if '回答' in k:
#                 answer_prob = v['true_probability']
#             elif 'トリプル' in k:
#                 match = re.search(r'トリプル(\d+)確信度', k)
#                 if match:
#                     triple_num = int(match.group(1))
#                     triple_probs[triple_num-1] = v['true_probability']
#         if debug:
#             print(f"triple_verb_1s_cot_is_true: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


def _run_triple_is_true_prob_family(method, question, templates, model_name, language, test_suffix, debug, get_response, recorder=None, test_instance=None):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt1s = templates["triple_verb_2s_top_1"].format(THE_QUESTION=question)
    if method == "triple_ling_2s_human7":
        prompt2s = templates["triple_ling_2s_human"].format(EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_7JP)
    else:
        prompt2s = templates[method].format(EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_JP)

    res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
    if not res.answers:
        return None
    else:
        answer = res.answers[0]
        triples = res.triple_sets[1].triples
        answer_prob = confidence.answer_conf[0]
        if confidence.triple_sets:
            triple_probs = confidence.triple_sets[1].triple_conf

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_verb_1s_top_1_family(method, question, templates, model_name, language, test_suffix, debug, get_response):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    if method == "triple_verb_1s_top_1_for_reasoning_model":
        response = get_response(prompt, max_tokens=4096)
    else:
        response = get_response(prompt)
    if not response:
        if debug:
            print('response is None, continue')
        return None
    res = parse_output(response)
    if not res.answers:
        if debug:
            print('res.answers is None, continue')
        return None
    else:
        answer = res.answers[0]
        if res.answer_conf:
            answer_prob = res.answer_conf[0]
        if res.triple_sets:
            triples = res.triple_sets[1].triples
            if res.triple_sets[1].triple_conf:
                triple_probs = res.triple_sets[1].triple_conf

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
    if answer_prob is None and not 'noconf' in test_suffix:
        if debug:
            print('answer_prob is None, continue')
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_verb_1s_top_family(method, question, templates, model_name, language, test_suffix, debug, get_response):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        return None
    res = parse_output(response)
    if not res.answers:
        return None
    else:
        answer = res.answers[0]
        triples = res.triple_sets[1].triples
        answer_prob = res.answer_conf[0]
        triple_probs = res.triple_sets[1].triple_conf

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
    if answer_prob is None:
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_verb_1s_cot_family(method, question, templates, model_name, language, test_suffix, debug, get_response):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt = templates[method].format(THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        return None
    res = parse_output(response)
    if debug:
        print(res)
    if not res.answers:
        return None
    else:
        answer = res.answers[0]
        if res.answer_conf:
            answer_prob = res.answer_conf[0]
        if res.triple_sets:
            triples = res.triple_sets[1].triples
            if res.triple_sets[1].triple_conf:
                triple_probs = res.triple_sets[1].triple_conf
            else:
                return None

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
    if answer_prob is None and not 'noconf' in test_suffix:
        if debug:
            print('answer_prob is None, continue')
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_verb_2s_family(method, question, templates, model_name, language, test_suffix, debug, get_response, recorder=None, test_instance=None):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    if method in ["triple_verb_2s_top_1", "triple_verb_2s_top_1_a"]:
        prompt1s = templates["triple_verb_2s_top_1"].format(THE_QUESTION=question)
        prompt2s = templates["triple_verb_2s_prob"]
        if method == "triple_verb_2s_top_1_a":
            prompt2s = templates["triple_verb_2s_prob_a"]

        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
        answer = '回答不可'
        answer_prob = None
        triples = []
        triple_probs = []
        if not res.answers:
            return None
        else:
            answer = res.answers[0]
        if not res.triple_sets:
            return None
        else:
            triples = res.triple_sets[1].triples
        if not confidence.answer_conf:
            return None
        else:
            answer_prob = confidence.answer_conf[0]
        if not confidence.triple_sets:
            return None
        else:
            triple_probs = confidence.triple_sets[1].triple_conf
        if debug:
            print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
        if answer_prob is None:
            return None

    elif "top_" in method:
        k = int(method.split("_")[-1])
        prompt_top = templates["triple_verb_2s_top_k"].format(THE_QUESTION=question, k=k)
        if debug:
            print(f"{method}: prompt:{prompt_top}")
        prob_prompt = templates["triple_verb_2s_top_k_prob"].format(k=k)

        res, confidence = run_two_step_evaluation(model_name, prompt_top, prob_prompt, debug=debug, recorder=recorder, test_instance=test_instance)
        answer = '回答不可'
        answer_prob = None
        triples = []
        triple_probs = []
        if not res.answers:
            return None
        else:
            if debug:
                print(f"{method}: res:{res},confidence:{confidence}")
            answer = res.answers[0]
            triples = res.triple_sets[1].triples
            answer_prob = confidence.answer_conf[0]
            triple_probs = confidence.triple_sets[1].triple_conf

            if debug:
                print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")

            if not answer_prob:
                return None

    elif method in ["triple_verb_2s_cot", "triple_verb_2s_cot_a"]:
        prompt1s = templates["triple_verb_2s_cot"].format(THE_QUESTION=question)
        prompt2s = templates["triple_verb_2s_prob"]
        if method == "triple_verb_2s_cot_a":
            prompt2s = templates["triple_verb_2s_prob_a"]
        if debug:
            print(f"prompt:{prompt1s}")

        res, confidence = run_two_step_evaluation(model_name, prompt1s, prompt2s, debug=debug, recorder=recorder, test_instance=test_instance)
        if not res.answers:
            answer = '回答不可'
            answer_prob = None
            triples = []
            triple_probs = []
            return None
        else:
            answer = res.answers[0]
            triples = res.triple_sets[1].triples
            answer_prob = confidence.answer_conf[0]
            if confidence.triple_sets:
                triple_probs = confidence.triple_sets[1].triple_conf
            else:
                triple_probs = [None] * len(triples)

            if debug:
                print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
            if answer_prob is None:
                return None

    else:
        print('ERROR! unknown method name:', method)

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def _run_triple_ling_1s_human_family(method, question, templates, model_name, language, test_suffix, debug, get_response):
    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    if language == 'en':
        prompt = templates[method].format(EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS, THE_QUESTION=question)
    else:
        if method == "triple_ling_1s_human7":
            prompt = templates["triple_ling_1s_human"].format(EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_7JP, THE_QUESTION=question)
        else:
            prompt = templates[method].format(EXPRESSION_LIST=LINGUISTIC_EXPRESSIONS_JP, THE_QUESTION=question)
    if debug:
        print(f"prompt:{prompt}")
    response = get_response(prompt)
    if not response:
        answer = '回答不可'
        answer_prob = None
        triples = []
        triple_probs = []
        return None
    res = parse_output(response)
    if debug:
        print(res)
    if not res.answers or not res.answer_conf:
        return None
    else:
        answer = res.answers[0]
        triples = res.triple_sets[1].triples
        answer_prob = res.answer_conf[0]
        triple_probs = res.triple_sets[1].triple_conf

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
    if answer_prob is None:
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


# def _run_triple_verb_3s_prob(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     if model_name == "gpt-4.1-2025-04-14":
#         prompt = templates["triple_verb_3s_top_2"].format(THE_QUESTION=question)
#     else:
#         prompt = templates["triple_verb_3s_top_1"].format(THE_QUESTION=question)

#     if debug:
#         print(f"prompt:{prompt}")
#     client = init_client()
#     messages = [
#         {"role": "system", "content": "あなたは質問に正確に答えるアシスタントです。"},
#         {"role": "user", "content": prompt}
#     ]
#     response1 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step1_response = response1.choices[0].message.content
#     res = parse_output(step1_response)
#     if debug:
#         print('step1_response', step1_response)
#         print('parse_output', res)
#     triples = res.triple_sets[1].triples
#     triple_probs = res.triple_sets[1].triple_conf

#     if debug:
#         print('triple_verb_3s: step1_response', step1_response)
#         print('triple_verb_3s: triples', triples)

#     messages.append({"role": "assistant", "content": step1_response})

#     messages.append({"role": "user", "content": templates['triple_verb_3s_triple_prob']})

#     response2 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step2_response = response2.choices[0].message.content
#     res = parse_output(step2_response)
#     triple_probs = res.triple_sets[1].triple_conf
#     if debug:
#         print('triple_verb_3s: triple_probs', triple_probs)

#     triple_and_confidence_list = []
#     for i, zip_p in enumerate(zip(triples, triple_probs)):
#         t, p = zip_p
#         triple_and_confidence_list.append(f"トリプル{i+1}: {t} - 確信度: {p}")
#     triple_and_confidence = '\n'.join(triple_and_confidence_list)
#     prompt3 = templates["triple_verb_3s_answer_prob"].format(THE_QUESTION=question, TRIPLE_AND_CONFIDENCES=triple_and_confidence)

#     messages.append({"role": "assistant", "content": step2_response})

#     messages.append({"role": "user", "content": prompt3})

#     response3 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step3_response = response3.choices[0].message.content
#     res = parse_output(step3_response)
#     if debug:
#         print('step3_response', step3_response)
#         print('res', res)
#     answer = res.answers[0]
#     answer_prob = res.answer_conf[0]

#     if debug:
#         print(f"triple_verb_3s: answer={answer}, answer_prob={answer_prob}")

#     if answer_prob is None:
#         return None

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


# def _run_triple_verb_3s(method, question, templates, model_name, language, test_suffix, debug, get_response, get_logprob_response=None):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt = templates["triple_verb_3s"].format(THE_QUESTION=question)
#     if debug:
#         print(f"triple_verb_3s prompt1:{prompt}")
#     client = init_client()
#     messages = [
#         {"role": "system", "content": "あなたは質問に正確に答えるアシスタントです。"},
#         {"role": "user", "content": prompt}
#     ]
#     response1 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step1_response = response1.choices[0].message.content
#     res = parse_output(step1_response)
#     if debug:
#         print('step1_response', step1_response)
#         print('parse_output', res)
#     triples = res.triple_sets[1].triples
#     triple_probs = res.triple_sets[1].triple_conf

#     if debug:
#         print('triple_verb_3s: step1_response', step1_response)
#         print('triple_verb_3s: triples', triples)

#     messages.append({"role": "assistant", "content": step1_response})

#     messages.append({"role": "user", "content": templates['triple_verb_3s_triple_prob']})

#     response2 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step2_response = response2.choices[0].message.content
#     res = parse_output(step2_response)
#     triple_probs = res.triple_sets[1].triple_conf
#     if debug:
#         print('triple_verb_3s: triple_probs', triple_probs)

#     triple_and_confidence_list = []
#     for i, zip_p in enumerate(zip(triples, triple_probs)):
#         t, p = zip_p
#         triple_and_confidence_list.append(f"トリプル{i+1}: {t} - 確信度: {p}")
#     triple_and_confidence = '\n'.join(triple_and_confidence_list)
#     prompt3 = templates["triple_verb_3s_answer_prob"].format(THE_QUESTION=question, TRIPLE_AND_CONFIDENCES=triple_and_confidence)

#     messages.append({"role": "assistant", "content": step2_response})

#     messages.append({"role": "user", "content": prompt3})

#     response3 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step3_response = response3.choices[0].message.content
#     res = parse_output(step3_response)
#     if debug:
#         print('step3_response', step3_response)
#         print('res', res)
#     answer = res.answers[0]
#     answer_prob = res.answer_conf[0]

#     if debug:
#         print(f"triple_verb_3s: answer={answer}, answer_prob={answer_prob}")

#     if answer_prob is None:
#         return None

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


# def _run_triple_verb_meta_family(method, question, templates, model_name, language, test_suffix, debug, get_response):
#     answer, triples, answer_prob, triple_probs = None, [], None, []
#     p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
#     logprobs_info = None

#     prompt = templates["triple_verb_1s_top_1"].format(THE_QUESTION=question)
#     if debug:
#         print(f"prompt:{prompt}")
#     client = init_client()
#     messages = [
#         {"role": "system", "content": "あなたは質問に正確に答えるアシスタントです。"},
#         {"role": "user", "content": prompt}
#     ]
#     response1 = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=0.0
#     )
#     step1_response = response1.choices[0].message.content
#     res = parse_output(step1_response)
#     if debug:
#         print('step1_response', step1_response, res)
#     if not res.answers:
#         return None
#     else:
#         answer = res.answers[0]
#         triples = res.triple_sets[1].triples
#         answer_prob = res.answer_conf[0]
#         triple_probs = res.triple_sets[1].triple_conf
#         if debug:
#             print(f"answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
#         if answer_prob is None:
#             return None
#         triple_and_confidence_list = []
#         for i, zip_p in enumerate(zip(triples, triple_probs)):
#             t, p = zip_p
#             triple_and_confidence_list.append(f"トリプル{i+1}: {t} - 確信度: {p}")
#         triple_and_confidence_list.append(f"最終回答: {answer} - 確信度: {answer_prob}")
#         triple_and_confidence = '\n'.join(triple_and_confidence_list)

#         prompt = templates["triple_verb_meta"].format(THE_QUESTION=question, ANSWER_TRIPLE_AND_CONFIDENCES=triple_and_confidence)
#         if debug:
#             print(f"prompt:{prompt}")

#         messages.append({"role": "assistant", "content": step1_response})

#         messages.append({"role": "user", "content": prompt})

#         response2 = client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             temperature=0.0
#         )
#         step2_response = response2.choices[0].message.content

#         res = parse_output(step2_response)
#         if debug:
#             print('step2_response', step2_response, res)
#         if not res.answer_conf:
#             return None
#         else:
#             answer_prob = res.answer_conf[0]
#             triple_probs = res.triple_sets[1].triple_conf
#             if debug:
#                 print(f"{method}: 調整後answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
#     if debug:
#         print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
#     if answer_prob is None:
#         return None

#     return EvidenceMethodResult(
#         answer, triples, answer_prob, triple_probs, logprobs_info,
#         p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
#     )


def _run_triple_budgeted_2step_baseline(method, question, templates, model_name, language, test_suffix, debug, get_response, recorder=None, test_instance=None):
    """
    Budgeted 2-step baseline for reasoning models (DeepSeek-R1, etc.)

    Turn0: Stream call with budget-based truncation (default: 100 tokens)
    Turn1: Final output with Turn0 reasoning as context
    """
    from finegrained_conf.llm.api_wrapper import run_budgeted_reasoning_2turn

    answer, triples, answer_prob, triple_probs = None, [], None, []
    p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
    logprobs_info = None

    prompt_start = templates["triple_budgeted_2step_start"].format(THE_QUESTION=question)
    final_prompt = templates["triple_budgeted_2step_final"]

    budget_tokens_reasoning = 100  # Reasoning token budget (default)
    max_tokens_reasoning_server = 4096  # Server-side max_tokens (set high)
    max_tokens_final = 1024  # Max tokens for the final turn
    temperature_reasoning = 0.0  # Temperature for the reasoning turn (0.0 recommended for stability)
    temperature_final = 0.0  # Temperature for the final turn

    if debug:
        print(f"Budgeted 2-step baseline method: {method}")
        print(f"prompt_start: {prompt_start}")
        print(f"final_prompt: {final_prompt}")
        print(f"budget_tokens_reasoning: {budget_tokens_reasoning}")

    # run_budgeted_reasoning_2turn を呼び出し
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
            stop_reasoning=["[OUTPUT]", "[/OUTPUT]"],
            stop_final=["[/OUTPUT]"],
            recorder=recorder,
            test_instance=test_instance,
            method_name=method,
            log_reasoning=True,
        )
    except Exception as e:
        if debug:
            print(f"Error in run_budgeted_reasoning_2turn: {e}")
        return None

    final_response = result["final_response_text"]

    if not final_response:
        if debug:
            print('final_response is None, continue')
        return None

    res = parse_output(final_response)

    if not res.answers:
        if debug:
            print('res.answers is None, continue')
        return None
    else:
        answer = res.answers[0]
        if res.answer_conf:
            answer_prob = res.answer_conf[0]
        if res.triple_sets:
            triples = res.triple_sets[1].triples
            if res.triple_sets[1].triple_conf:
                triple_probs = res.triple_sets[1].triple_conf

    if debug:
        print(f"{method}: answer:{answer}, answer_prob:{answer_prob}, triples:{triples}, triple_probs:{triple_probs}")
        print(f"Turn0 estimated tokens: {result['turn0_est_tokens']}")

    if answer_prob is None and 'noconf' not in test_suffix:
        if debug:
            print('answer_prob is None, continue')
        return None

    return EvidenceMethodResult(
        answer, triples, answer_prob, triple_probs, logprobs_info,
        p_hat_ans, p_hat_evset, p_hat_trip, set_prob, H_nat_ans, H_nat_evset, H_nat_trip
    )


def execute_evidence_method(
    method: str,
    model_name: str,
    question: str,
    templates: Dict[str, str],
    language: str,
    test_suffix: str,
    debug: bool,
    get_response,
    get_logprob_response=None,
    recorder=None,
    test_instance=None,
):
    direct_handlers = {
        "triple_label_prob": _run_triple_label_prob,
        "triple_logprob": _run_triple_logprob,
        "triple_logprob_cot": _run_triple_logprob_cot,
        # "triple_is_true_logprob": _run_triple_is_true_logprob,
        # "triple_is_true_cot1s_logprob": _run_triple_is_true_cot1s_logprob,
        # "triple_is_true_cot2s_logprob": _run_triple_is_true_cot2s_logprob,
        # "triple_verb_1s_cot_is_true": _run_triple_verb_1s_cot_is_true,
        # "triple_verb_3s_prob": _run_triple_verb_3s_prob,
        # "triple_verb_3s": _run_triple_verb_3s,
        # "triple_verb_long_reasoning": _run_triple_verb_long_reasoning,
        "triple_budgeted_2step_baseline": _run_triple_budgeted_2step_baseline,
    }

    is_true_prob_methods = [
        "triple_is_true_prob",
        "triple_is_true_prob_a",
        "triple_ling_2s_human",
        "triple_ling_2s_human_a",
        "triple_ling_2s_human7",
    ]
    verb_1s_top_1_methods = [
        "triple_verb_1s_top_1",
        "triple_verb_1s_top_1_a",
        "triple_verb_1s_top_1_ansconf",
        "triple_verb_1s_top_1_noconf",
        "triple_verb_1s_top_1_for_reasoning_model"
    ]
    verb_1s_cot_methods = [
        "triple_verb_1s_cot",
        "triple_verb_1s_cot_a",
        "triple_verb_1s_cot_ansconf",
        "triple_verb_1s_cot_noconf",
        "triple_cot_level_baseline",
    ]
    ling_1s_methods = ["triple_ling_1s_human", "triple_ling_1s_human_a", "triple_ling_1s_human7"]

    if method in direct_handlers:
        handler = direct_handlers[method]
        return handler(
            method,
            question,
            templates,
            model_name,
            language,
            test_suffix,
            debug,
            get_response,
            get_logprob_response,
            recorder,
            test_instance,
        )
    if method in is_true_prob_methods:
        return _run_triple_is_true_prob_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response, recorder, test_instance
        )
    if method in verb_1s_top_1_methods:
        return _run_triple_verb_1s_top_1_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response
        )
    if method.startswith("triple_verb_1s_top_"):
        return _run_triple_verb_1s_top_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response
        )
    if method in verb_1s_cot_methods:
        return _run_triple_verb_1s_cot_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response
        )
    if method.startswith("triple_verb_2s_"):
        return _run_triple_verb_2s_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response, recorder, test_instance
        )
    if method in ling_1s_methods:
        return _run_triple_ling_1s_human_family(
            method, question, templates, model_name, language, test_suffix, debug, get_response
        )
    # if "triple_verb_meta" in method:
    #     return _run_triple_verb_meta_family(
    #         method, question, templates, model_name, language, test_suffix, debug, get_response
    #     )

    raise IncompleteResponseError(f"Unknown method: {method}")

# 主要なメソッドを実行し、結果を収集する関数
def run_experiment_triple(
    model_name,
    dataset_name,
    split,
    methods,
    num_samples=100,
    language="en",
    test_suffix="",
    overwrite=True,
    overwrite_eval=False,
    debug=False,
    recorder: Optional[ExperimentRecorder] = None,
    current_test_instance: Optional[TestInstance] = None,
):
    """
    実験を実行し、結果を収集する
    
    Args:
        model_name: 使用するモデル名
        dataset_name: 使用するデータセット名
        methods: 実行するメソッドのリスト
        num_samples: 使用するサンプル数
        language: 使用する言語（"en"または"ja"）
        
    Returns:
        各メソッドの結果を含む辞書
    """
    # データセットをロード
    dataset = load_dataset(dataset_name, split=split, max_samples=num_samples)
    api_wrapper = APIWrapper(recorder=recorder)
    if debug:
        print(dataset_name, split, num_samples)
        for i in range(min(3, len(dataset))):
            sample = dataset.get_sample(i)
            # print(sample)
            print(f"  質問: {sample['question']}")
            print(f"  回答: {sample['answer']}")
            if 'derivations' in sample:
                print(f"  導出: {sample['derivations']}")
            print()
    
    # 言語に応じたプロンプトテンプレートとマッピングを選択
    if language == "ja":
        templates = TRIPLE_PROMPT_TEMPLATES_JP
        linguistic_to_prob = LING_JA
        # linguistic_expressions = LINGUISTIC_EXPRESSIONS_JP
    else:
        templates = TRIPLE_PROMPT_TEMPLATES_EN
        linguistic_to_prob = LING_EN
        # linguistic_expressions = LINGUISTIC_EXPRESSIONS
        # evaluation_prompt_template = TRIPLE_EVALUATION_PROMPT            
    
    results = {}
    tmp_result_file = f"tmp_results/results_triples_{model_name}_{dataset_name}_{split}{test_suffix}_{num_samples}.json"
    
    for method in methods:
        print(f"Running method for triple: {method} on {model_name} with {dataset_name} in {language}")
        
        tmp_method_result_file = f"tmp_results/results_triples_{model_name}_{dataset_name}_{split}_{method}{test_suffix}_{num_samples}.json"

        if not overwrite and os.path.isfile(tmp_method_result_file):
            LOAD_RESULT = True
            with open(tmp_method_result_file, 'r', encoding='utf-8') as f:
                print(f"load tmp_method_result_file : {tmp_method_result_file}")
                data = json.load(f)
                # if method == "triple_logprob":
                if method in ["triple_logprob", "triple_is_true_cot1s_logprob", "triple_logprob_cot"]:
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
            LOAD_RESULT = False
            answers = []
            triples_list = []
            answers_probs = []
            triples_probs_list = []
            answer_correctness = []
            triples_correctness_list = []
            gold_triples_correctness_list = []
            logprobs_info_list = []
            logprobs_info = None
            p_hat_ans_list, p_hat_evset_list, p_hat_trip_list, set_prob_list, H_nat_ans_list, H_nat_evset_list, H_nat_trip_list = [], [], [], [], [], [], []

            # Use JSONL format for intermediate results (one line per question)
            tmp_qa_result_file = f"tmp_results/results_triples_qa_{model_name}_{dataset_name}_{split}_{method}{test_suffix}_{num_samples}.jsonl"

            # Load existing results by question_id for resumption
            processed_question_ids = {}
            if not overwrite and os.path.isfile(tmp_qa_result_file):
                print(f"load tmp_qa_result_file : {tmp_qa_result_file}")
                with open(tmp_qa_result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            qid = entry.get("question_id")
                            if qid is not None:
                                processed_question_ids[qid] = entry

                # Reconstruct arrays in original order for backward compatibility
                # This ensures the final output format remains consistent
                for i in range(len(dataset)):
                    sample = dataset.get_sample(i)
                    question_id = str(sample.get("question_id", sample.get("qid", i)))

                    if question_id in processed_question_ids:
                        entry = processed_question_ids[question_id]
                        answers.append(entry.get("answer"))
                        answers_probs.append(entry.get("answer_confidence"))
                        answer_correctness.append(entry.get("answer_correctness"))
                        triples_list.append(entry.get("triples"))
                        triples_probs_list.append(entry.get("org_triple_confidences"))
                        triples_correctness_list.append(entry.get("org_triple_correctness"))
                        gold_triples_correctness_list.append(entry.get("org_gold_triple_correctness"))
                        if "p_hat_ans" in entry:  p_hat_ans_list.append(entry.get("p_hat_ans"))
                        if "p_hat_evset" in entry:  p_hat_evset_list.append(entry.get("p_hat_evset"))
                        if "p_hat_trip" in entry:  p_hat_trip_list.append(entry.get("p_hat_trip"))
                        if "set_prob" in entry:  set_prob_list.append(entry.get("set_prob"))
                        if "H_nat_ans" in entry:  H_nat_ans_list.append(entry.get("H_nat_ans"))
                        if "H_nat_evset" in entry:  H_nat_evset_list.append(entry.get("H_nat_evset"))
                        if "H_nat_trip" in entry:  H_nat_trip_list.append(entry.get("H_nat_trip"))

            for i in tqdm(range(len(dataset))):
                sample = dataset.get_sample(i)
                question_index = sample.get("question_index", i)
                if question_index is None:
                    question_index = i
                question_id = str(sample.get("question_id", sample.get("qid", question_index)))
                dataset_native_id = str(sample.get("dataset_native_id", sample.get("qid", question_id)))

                # Skip if already processed
                if question_id in processed_question_ids:
                    continue

                question = sample["question"]
                gold_answer = sample["answer"]
                gold_triples = sample['derivations']

                current_test_instance = None
                if recorder is not None:
                    current_test_instance = TestInstance(
                        dataset=dataset_name,
                        split=split,
                        question_index=question_index,
                        question_id=question_id,
                        dataset_native_id=dataset_native_id,
                    )

                if debug:
                    print(f"question:{question}, gold_answer:{gold_answer}, gold_triples:{gold_triples}")

                def get_response(prompt: str, **kwargs):
                    response = api_wrapper.call_llm(
                        lambda: get_model_response(
                            model_name,
                            prompt,
                            method_name=method,
                            recorder=recorder,
                            test_instance=current_test_instance,
                            **kwargs,
                        ),
                        current_test_instance,
                        method,
                        "answer_generation",
                        sample_index=0,
                    )
                    if debug:
                        print(f"response:{response}")

                    return response

                def get_logprob_response(prompt: str, **kwargs):
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
                answer = None
                triples, answer_prob, triple_probs = [], None, []
                p_hat_ans = p_hat_evset = p_hat_trip = set_prob = H_nat_ans = H_nat_evset = H_nat_trip = None
                logprobs_info = None
                result = execute_evidence_method(
                    method,
                    model_name,
                    question,
                    templates,
                    language,
                    test_suffix,
                    debug,
                    get_response,
                    get_logprob_response,
                    recorder,
                    current_test_instance,
                )

                # If no response was obtained due to content filtering or other reasons.
                if result is None:
                    print('WARNING: result is None, skipping question:', question, gold_answer, gold_triples)
                    triples_is_correct, gold_triples_is_correct = [], []
                    answer_is_correct = None
                    answers.append(None)
                    triples_list.append([])
                    answers_probs.append(None)
                    triples_probs_list.append([])
                    answer_correctness.append(0.0)
                    triples_correctness_list.append(triples_is_correct)
                    gold_triples_correctness_list.append(gold_triples_is_correct)
                    if method == "triple_label_prob":
                        p_hat_ans_list.append(None)
                        p_hat_evset_list.append(None)
                        p_hat_trip_list.append(None)
                        set_prob_list.append(None)
                        H_nat_ans_list.append(None)
                        H_nat_evset_list.append(None)
                        H_nat_trip_list.append(None)
                    continue

                answer = result.answer
                triples = result.triples
                answer_prob = result.answer_prob
                triple_probs = result.triple_probs
                logprobs_info = result.logprobs_info
                p_hat_ans = result.p_hat_ans
                p_hat_evset = result.p_hat_evset
                p_hat_trip = result.p_hat_trip
                set_prob = result.set_prob
                H_nat_ans = result.H_nat_ans
                H_nat_evset = result.H_nat_evset
                H_nat_trip = result.H_nat_trip

                if not answer or not triples:
                    print('WARNING: skip question:', question, gold_answer, gold_triples, answer, triples)
                    triples_is_correct, gold_triples_is_correct = [], []
                    answer_is_correct = None
                    answers.append(answer)
                    triples_list.append(triples)
                    answers_probs.append(answer_prob)
                    triples_probs_list.append(triple_probs)
                    answer_correctness.append(1.0 if answer_is_correct else 0.0)
                    triples_correctness_list.append(triples_is_correct)
                    gold_triples_correctness_list.append(gold_triples_is_correct)
                    if method == "triple_label_prob":
                        p_hat_ans_list.append(p_hat_ans)
                        p_hat_evset_list.append(p_hat_evset)
                        p_hat_trip_list.append(p_hat_trip)
                        set_prob_list.append(set_prob)
                        H_nat_ans_list.append(H_nat_ans)
                        H_nat_evset_list.append(H_nat_evset)
                        H_nat_trip_list.append(H_nat_trip)

                    continue

                # Check whether the answer is correct.
                answer_is_correct = check_answer_correctness(
                    answer, gold_answer, question, model="gpt-4.1-2025-04-14", language=language, debug=debug
                )
                if triples:
                    triples_is_correct, gold_triples_is_correct = check_triples_correctness(triples, gold_triples, question, model="gpt-4.1-2025-04-14", language=language, debug=debug)

                else:
                    # triples_is_correct = []
                    # gold_triples_is_correct = [0.0] * len(gold_triples)
                    continue

                answers.append(answer)
                triples_list.append(triples)
                answers_probs.append(answer_prob)
                triples_probs_list.append(triple_probs)
                answer_correctness.append(1.0 if answer_is_correct else 0.0)
                triples_correctness_list.append(triples_is_correct)
                gold_triples_correctness_list.append(gold_triples_is_correct)
                if logprobs_info:
                    logprobs_info_list.append(logprobs_info)
                if method == "triple_label_prob":
                    p_hat_ans_list.append(p_hat_ans)
                    p_hat_evset_list.append(p_hat_evset)
                    p_hat_trip_list.append(p_hat_trip)
                    set_prob_list.append(set_prob)
                    H_nat_ans_list.append(H_nat_ans)
                    H_nat_evset_list.append(H_nat_evset)
                    H_nat_trip_list.append(H_nat_trip)

                # Save individual question result to JSONL (append mode)
                question_result = {
                    "question_id": question_id,
                    "question_index": question_index,
                    "dataset_native_id": dataset_native_id,
                    "answer": answer,
                    "answer_confidence": answer_prob,
                    "answer_correctness": 1.0 if answer_is_correct else 0.0,
                    "triples": triples,
                    "org_triple_confidences": triple_probs,
                    "org_triple_correctness": triples_is_correct,
                    "org_gold_triple_correctness": gold_triples_is_correct,
                }
                if method == "triple_label_prob":
                    question_result["p_hat_ans"] = p_hat_ans
                    question_result["p_hat_evset"] = p_hat_evset
                    question_result["p_hat_trip"] = p_hat_trip
                    question_result["set_prob"] = set_prob
                    question_result["H_nat_ans"] = H_nat_ans
                    question_result["H_nat_evset"] = H_nat_evset
                    question_result["H_nat_trip"] = H_nat_trip

                # Append to JSONL file (one line per question)
                os.makedirs(os.path.dirname(tmp_qa_result_file) or ".", exist_ok=True)
                with open(tmp_qa_result_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(question_result, ensure_ascii=False) + "\n")
                
                # break # break retry
                
            if not answers_probs:
                continue

            def format_answer(answers, triples_list, answers_probs, triples_probs_list, answer_correctness, triples_correctness_list, gold_triples_correctness_list):
                print(answers, triples_list, answers_probs, triples_probs_list, answer_correctness, triples_correctness_list)
                # flattened_triples_probs = [element for sublist in triples_probs_list for element in sublist if element else 0.5]
                flattened_triples_probs = [
                    (elem if elem is not None else np.nan)
                    for sublist in triples_probs_list
                    for elem    in sublist
                ]
                # flattened_triples_correctness = [element for sublist in triples_correctness_list for element in sublist if element else 0.0]
                flattened_triples_correctness = [
                    (elem if elem is not None else np.nan)
                    for sublist in triples_correctness_list
                    for elem    in sublist
                ]
                print('answers_probs', answers_probs, type(answers_probs))
                print('answer_correctness', answer_correctness, type(answer_correctness))
                print('flattened_triples_probs', flattened_triples_probs, type(flattened_triples_probs))
                print('flattened_triples_correctness', flattened_triples_correctness, type(flattened_triples_correctness))

                res = {
                    "answers": answers,
                    "answer_confidences": answers_probs,
                    "answer_correctness": answer_correctness,
                    "triples": triples_list,
                    "triple_confidences": flattened_triples_probs,
                    "org_triple_confidences": triples_probs_list,
                    "triple_correctness": flattened_triples_correctness,
                    "org_triple_correctness": triples_correctness_list,
                    "org_gold_triple_correctness": gold_triples_correctness_list,
                    "answer_metrics": None,
                    "triple_metrics": None
                }
                return res
            
            # if method in ["triple_logprob", "triple_is_true_logprob", "triple_is_true_cot1s_logprob"]:
            if method in ["triple_logprob", "triple_is_true_cot1s_logprob", "triple_logprob_cot"]:
                tmp_result = {}
                answers_prob_dict = defaultdict(list)
                triple_probs_list_dict = defaultdict(list)
                probs_keys = ["mean_logprob", "sum_logprob", "prod_prob", "mean_prob", "min_prob", "normalized_prod_prob", "length_penalized_prob", "linear_scaled_prob"]
                for a_dic in answers_probs:
                    if not a_dic:
                        for k in probs_keys:
                            answers_prob_dict[k].append(None)
                    else:
                        for k, v in a_dic.items():
                            answers_prob_dict[k].append(v)
                print('triples_probs_list', triples_probs_list)
                for t_list in triples_probs_list:
                    print('t_list', t_list)
                    tmp_triple_probs_list_dict = defaultdict(list)
                    for t_dic in t_list: 
                        for k, v in t_dic.items():
                            tmp_triple_probs_list_dict[k].append(v)
                    print('tmp_triple_probs_list_dict', tmp_triple_probs_list_dict)
                    for k, v in tmp_triple_probs_list_dict.items():
                        triple_probs_list_dict[k].append(v)
                    print('triple_probs_list_dict', triple_probs_list_dict)
                for k, v in answers_prob_dict.items():
                    if k.endswith('prob'):
                        if method in k:
                            tmp_method_name = k
                        else:
                            tmp_method_name = method+"_"+k
                        tmp_res = format_answer(answers, triples_list, v, triple_probs_list_dict[k], answer_correctness, triples_correctness_list, gold_triples_correctness_list)
                        if method == "triple_label_prob":
                            tmp_res["p_hat_ans"] = p_hat_ans_list
                            tmp_res["p_hat_evset"] = p_hat_evset_list
                            tmp_res["p_hat_trip"] = p_hat_trip_list
                            tmp_res["set_prob"] = set_prob_list
                            tmp_res["H_nat_ans"] = H_nat_ans_list
                            tmp_res["H_nat_evset"] = H_nat_evset_list
                            tmp_res["H_nat_trip"] = H_nat_trip_list
                        tmp_result[tmp_method_name] = tmp_res


            else:
                res = format_answer(answers, triples_list, answers_probs, triples_probs_list, answer_correctness, triples_correctness_list, gold_triples_correctness_list)
                if method == "triple_label_prob":
                    res["p_hat_ans"] = p_hat_ans_list
                    res["p_hat_evset"] = p_hat_evset_list
                    res["p_hat_trip"] = p_hat_trip_list
                    res["set_prob"] = set_prob_list
                    res["H_nat_ans"] = H_nat_ans_list
                    res["H_nat_evset"] = H_nat_evset_list
                    res["H_nat_trip"] = H_nat_trip_list
                tmp_result = {method:res}

            # Save the results.
            with open(tmp_method_result_file, "w") as f:
                json.dump(tmp_result, f, ensure_ascii=False)
            
            if not LOAD_RESULT:
                with open(tmp_method_result_file.replace('.json', '_fixeval_0529.json'), "w") as f:
                    json.dump(tmp_result, f, ensure_ascii=False)
            
            # if logprobs_info_list:
            #     with open(tmp_method_result_file.replace('.json', '_logprobs.json'), "w") as f:
            #         json.dump({method:res}, f, ensure_ascii=False)
        
            
        # if method == "triple_logprob":
        for k, v in tmp_result.items():
            if len(v["answer_confidences"]) == len(v["answer_correctness"]) :
                v["answer_metrics"] = compute_metrics(v["answer_confidences"], v["answer_correctness"], model_name==k)
                conf_aligned, corr_aligned = align_conf_corr(v["org_triple_confidences"], v["org_triple_correctness"])

                v["triple_metrics"] = compute_metrics(conf_aligned, corr_aligned, model_name==k)
                results[k] = v

        with open(tmp_result_file, "w") as f:
            json.dump(results, f, ensure_ascii=False)
    
    return results


# for test
def main():
    parser = argparse.ArgumentParser()
    model_name = "gpt-4o-mini-2024-07-18"
    # model_name = "gpt-4o-2024-11-20"
    dataset_name = "jemhop_qa"
    methods =[
        "triple_logprob",
    ]
    # methods = ["verb_1s_top_1"]

    split = "dev"
    language = "ja"
    num_samples=120

    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_id = build_run_id(
        dataset=dataset_name,
        split=split,
        model=model_name,
        methods=methods,
        timestamp=timestamp,
    )
    recorder = ExperimentRecorder(
        run_id=run_id,
        model=model_name,
        dataset=dataset_name,
        split=split,
    )

    results = run_experiment_triple(
        model_name,
        dataset_name,
        split,
        methods,
        num_samples=num_samples,
        language=language,
        debug=True,
        recorder=recorder,
    )
    
    with open(f"results_triples_{model_name}_{dataset_name}_{split}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)

    summary = {}
    for method, result in results.items():
        summary[method] = {
            "answer_accuracy": np.mean(result["answer_correctness"]),
            "answer_ece": result["answer_metrics"]["ece"],
            "answer_ece_t": result["answer_metrics"]["ece_t"],
            "answer_auc": result["answer_metrics"]["auc"],
            "triple_accuracy": np.mean(result["triple_correctness"]),
            "triple_ece": result["triple_metrics"]["ece"],
            "triple_ece_t": result["triple_metrics"]["ece_t"],
            "triple_auc": result["triple_metrics"]["auc"]
        }
    
    print(json.dumps(summary, indent=2, ensure_ascii=False))