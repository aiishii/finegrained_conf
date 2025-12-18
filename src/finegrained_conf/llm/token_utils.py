import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import re
import unicodedata

import re
import unicodedata
import logging
import bisect
import math
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_prefix_offsets(tokens: List[str]) -> List[int]:
    """
    各トークンの開始文字オフセットを累積して返す。
    offsets[i] は tokens[i] の先頭文字が連結文字列上で何文字目かを示す。
    """
    offsets = [0]
    for tok in tokens:
        offsets.append(offsets[-1] + len(tok))
    return offsets


def find_token_span(tokens: List[str], offsets: List[int], substring: str) -> Tuple[int, int]:
    """
    連結した tokens から substring の開始・終了トークンインデックスを返す。
    """
    joined = ''.join(tokens)
    start_char = joined.find(substring)
    if start_char < 0:
        raise ValueError(f"部分文字列 {substring!r} が見つかりませんでした。")
    end_char = start_char + len(substring) - 1

    start_tok = bisect.bisect_right(offsets, start_char) - 1
    end_tok = bisect.bisect_right(offsets, end_char) - 1
    return start_tok, end_tok


def extract_spans(tokens: List[str], forcing_match_answer=False, language="ja") -> Dict[str, Any]:
    """
    tokens から回答部とトリプル部のトークン範囲を抽出する。
    戻り値: {'answer': {...}, 'triples': [{...}, ...]}
    """
    spans: Dict[str, Any] = {}
    full_text = ''.join(tokens)

    if language == "ja":
        answer_match = re.search(r"回答:\s*(.+)", full_text)
    else:
        answer_match = re.search(r"Answer:\s*(.+)", full_text)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    if forcing_match_answer and not answer_text:
        spans['answer'] = {
                'start': 0,
                'end': len(tokens)-1,
                'tokens': tokens
            }
        spans['triples'] = []
        return spans

    if language == "ja":
        triple_iter = list(re.finditer(r"トリプル(\d+):\s*(\(.+?\))", full_text))
    else:
        triple_iter = list(re.finditer(r"Triple\s*(\d+):\s*(\(.+?\))", full_text))
    triples = [(int(m.group(1)), m.group(2)) for m in triple_iter]

    offsets = compute_prefix_offsets(tokens)


    if answer_text:
        try:
            a_start, a_end = find_token_span(tokens, offsets, answer_text)
            spans['answer'] = {
                'start': a_start,
                'end': a_end,
                'tokens': tokens[a_start:a_end+1]
            }
        except ValueError as e:
            logger.warning(str(e))

    spans['triples'] = []
    for num, txt in triples:
        try:
            t_start, t_end = find_token_span(tokens, offsets, txt)
            spans['triples'].append({
                'number': num,
                'start': t_start,
                'end': t_end,
                'tokens': tokens[t_start:t_end+1]
            })
        except ValueError as e:
            logger.warning(f"トリプル{num}: {e}")

    return spans


def normalize_text(text: str) -> str:
    """
    テキストを Unicode 正規化し、空白やカッコ、句読点を除去する。
    """
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r"[\s\n\(\)\[\]{}、，。,.]", '', text)


def process_byte_sequences(tokens: List[str]) -> Tuple[List[str], List[int]]:
    """
    エスケープされたバイトシーケンスをまとめて UTF-8 デコードし、
    デコード後トークンと元インデックスを返す。
    """
    processed: List[str] = []
    indices: List[int] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if '\\x' in tok:
            bytebuf = bytearray()
            start = i
            while i < len(tokens) and '\\x' in tokens[i]:
                try:
                    part = tokens[i].encode('ascii').decode('unicode-escape').encode('latin1')
                    bytebuf.extend(part)
                except Exception:
                    break
                i += 1
            try:
                decoded = bytebuf.decode('utf-8')
                processed.append(decoded)
                indices.append(start)
            except UnicodeDecodeError:
                for j in range(start, i):
                    processed.append(tokens[j])
                    indices.append(j)
        else:
            processed.append(tok)
            indices.append(i)
            i += 1
    return processed, indices


def calculate_token_confidences(
    text: str,
    tokens: List[str],
    token_indices: List[int],
    token_probs: List[float]
) -> Dict[str, Any]:
    """
    テキストに対してトークンの確信度を計算する。
    元の calculate_token_confidences の詳細メトリクスを再現。
    """
    norm_text = normalize_text(text)

    norm_tokens = [normalize_text(t) for t in tokens]

    filtered = [
        (tok, idx, prob, norm_tok)
        for tok, idx, prob, norm_tok in zip(tokens, token_indices, token_probs, norm_tokens)
        if norm_tok
    ]
    if filtered:
        filtered_tokens, filtered_indices, filtered_probs, filtered_norms = zip(*filtered)
    else:
        filtered_tokens, filtered_indices, filtered_probs, filtered_norms = [], [], [], []

    combined = ''.join(filtered_norms)
    coverage = len(combined) / len(norm_text) if norm_text else 0.0

    n_tokens = len(filtered_probs)
    mean_logprob = sum(filtered_probs) / n_tokens if n_tokens else 0.0
    sum_logprob = sum(filtered_probs)
    prod_prob = math.exp(sum_logprob)
    mean_prob = math.exp(mean_logprob)
    min_prob = math.exp(min(filtered_probs)) if filtered_probs else 0.0

    alpha = 0.6
    length_penalty = ((5 + n_tokens) ** alpha) / ((5 + 1) ** alpha)
    length_penalized_prob = math.exp(sum_logprob / length_penalty) if n_tokens else 0.0

    normalized_prod_prob = math.exp(sum_logprob / n_tokens) if n_tokens else 0.0

    beta = 0.1
    linear_scaled_prob = prod_prob * (1 + beta * n_tokens)

    return {
        "tokens": list(filtered_tokens),
        "token_indices": list(filtered_indices),
        "n_tokens": n_tokens,
        "logprobs": list(filtered_probs),
        "coverage": coverage,
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "prod_prob": prod_prob,
        "mean_prob": mean_prob,
        "min_prob": min_prob,
        "normalized_prod_prob": normalized_prod_prob,
        "length_penalized_prob": length_penalized_prob,
        "linear_scaled_prob": linear_scaled_prob
    }



def calculate_all_triple_confidences(
    response_tokens: List[str],
    response_probs: List[float],
    forcing_match_answer=False,
    language = "ja",
) -> Dict[str, Any]:
    """
    応答トークンと確率から、各トリプルのテキストと確信度を計算して返す。
    """
    proc_tokens, orig_indices = process_byte_sequences(response_tokens)

    spans = extract_spans(proc_tokens, forcing_match_answer=forcing_match_answer, language=language)
    results = {'answer': '', 'triples': []}

    t = spans.get('answer', {})
    if t:
        start, end = t['start'], t['end']
        tok_slice = t['tokens']
        idxs = [orig_indices[i] for i in range(start, end+1)]
        probs = [response_probs[i] for i in idxs]
        text = ''.join(tok_slice)

        conf = calculate_token_confidences(text, tok_slice, idxs, probs)
        results['answer'] = {
            'text': text,
            'confidence': conf
        }
    
    for t in spans.get('triples', []):
        num = t['number']
        start, end = t['start'], t['end']
        tok_slice = t['tokens']
        idxs = [orig_indices[i] for i in range(start, end+1)]
        probs = [response_probs[i] for i in idxs]
        text = ''.join(tok_slice)

        conf = calculate_token_confidences(text, tok_slice, idxs, probs)
        results['triples'].append({
            'num' : num,
            'text': text,
            'confidence': conf
        })

    return results

def test_with_complex_case():
    text = "(Apple，本社所在地，カリフォルニア州クパチーノ)"
    tokens = ['回答', ':', ' ア', 'メリ', 'カ', '合', '\\xe8\\xa1', '\\x86', '国', 'カ', 'リ', 'フォ', 'ル', 'ニ', 'ア', '州', 'ク', 'パ', 'チ', '\\xe3\\x83\\xbc\\xe3\\x83', '\\x8e', ' \n', 'ト', 'リ', 'プ', 'ル', '1', ':', ' (', 'i', 'Pod', '，', '製', '造', '，', 'Apple', ')', ' \n', 'ト', 'リ', 'プ', 'ル', '2', ':', ' (', 'Apple', '，本', '社', '所在地', '，', 'カ', 'リ', 'フォ', 'ル', 'ニ', 'ア', '州', 'ク', 'パ', 'チ', '\\xe3\\x83\\xbc\\xe3\\x83', '\\x8e', ')', ' ']
    token_probs = [-0.1] * len(tokens)

    for i, token in enumerate(tokens):
        if token == 'Apple':
            token_probs[i] = -0.01
        elif token == '，本':
            token_probs[i] = -0.02
        elif token == '社':
            token_probs[i] = -0.03
        elif token == '所在地':
            token_probs[i] = -0.04
        elif token == 'カ':
            token_probs[i] = -0.05
        elif token == 'リ':
            token_probs[i] = -0.06
        elif token == 'フォ':
            token_probs[i] = -0.07
        elif token == 'ル':
            token_probs[i] = -0.08
        elif token == 'ニ':
            token_probs[i] = -0.09
        elif token == 'ア':
            token_probs[i] = -0.10
        elif token == '州':
            token_probs[i] = -0.11
        elif token == 'ク':
            token_probs[i] = -0.12
        elif token == 'パ':
            token_probs[i] = -0.13
        elif token == 'チ':
            token_probs[i] = -0.14

    # result = calculate_token_confidences(text, tokens, token_probs)
    result = calculate_all_triple_confidences(tokens, token_probs)
    print(result)

    print("\n最終的な確信度計算結果:")
    for key, value in result.items():
        if key not in ["tokens", "token_indices", "logprobs"]:
            print(f"{key}: {value}")

# def calculate_normalized_token_logprobs(text: str, logprobs_info) -> dict:
#     """
    
#     Args:
        
#     Returns:
#     """
#     tokens = logprobs_info["tokens"]
#     token_logprobs = logprobs_info["logprobs"]

#     filtered_indices, filtered_tokens, filtered_logprobs = find_matching_tokens_improved(text, tokens, token_logprobs)
    
#     n_tokens = len(filtered_logprobs)
    
#     if filtered_logprobs and n_tokens > 0:
        
        
#         geometric_mean_prob = mean_prob
        
#         normalized_prod_prob = np.exp(sum_logprob / n_tokens)
        
#         length_penalty = ((5 + n_tokens)**alpha) / ((5 + 1)**alpha)
#         length_penalized_prob = np.exp(sum_logprob / length_penalty)
        
#         linear_scaled_prob = prod_prob * (1 + beta * n_tokens)
        
#         return {
#             "tokens": filtered_tokens,
#             "n_tokens": n_tokens,
#             "logprobs": filtered_logprobs,
#             "mean_logprob": mean_logprob,
#             "sum_logprob": sum_logprob,
#             "prod_prob": prod_prob,
#             "mean_prob": mean_prob,
#             "min_prob": min_prob,
            
#             "geometric_mean_prob": geometric_mean_prob,
#             "normalized_prod_prob": normalized_prod_prob,
#             "length_penalized_prob": length_penalized_prob,
#             "linear_scaled_prob": linear_scaled_prob
#         }
    
#     return {
#         "tokens": filtered_tokens,
#         "n_tokens": n_tokens,
#         "logprobs": filtered_logprobs,
#         "mean_logprob": None,
#         "sum_logprob": None,
#         "prod_prob": None,
#         "mean_prob": None,
#         "min_prob": None,
#         "geometric_mean_prob": None,
#         "normalized_prod_prob": None,
#         "length_penalized_prob": None,
#         "linear_scaled_prob": None
#     }