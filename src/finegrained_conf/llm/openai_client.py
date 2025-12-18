import openai
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import json
import os
# from extracter import *
from typing import List, Dict, Tuple, Optional
import math
import re
from finegrained_conf.utils.parser import parse_output
from finegrained_conf.io.run_metadata import ExperimentRecorder, TestInstance
from finegrained_conf.config import load_llm_config, get_model_config
from finegrained_conf.config.llm_config import apply_proxy_settings

# Load configuration once at module level
_llm_config = None

def _get_llm_config():
    """Get or load LLM configuration."""
    global _llm_config
    if _llm_config is None:
        try:
            # Enable verbose mode with LLM_CONFIG_VERBOSE=1 environment variable
            verbose = os.environ.get('LLM_CONFIG_VERBOSE', '0') == '1'
            _llm_config = load_llm_config(verbose=verbose)
            # Apply proxy settings if configured
            apply_proxy_settings(_llm_config)
            if verbose:
                print(f"[LLM Config] Configuration loaded successfully")
                print(f"[LLM Config] Has default config: {'default' in _llm_config}")
                print(f"[LLM Config] Has models config: {'models' in _llm_config}")
        except Exception as e:
            # Fallback to environment variables if config file not found
            print(f"Warning: Could not load LLM config file: {e}")
            print("Falling back to environment variables...")
            _llm_config = {}
    return _llm_config

def azure_client_init(model_name: Optional[str] = None):
    """
    Initialize Azure OpenAI client with configuration from llm_config.yaml.

    Args:
        model_name: Name of the model to get specific configuration for

    Returns:
        AzureOpenAI client instance
    """
    config = _get_llm_config()
    model_config = get_model_config(model_name, config)

    # Get configuration values
    api_key = model_config.get('api_key')
    azure_endpoint = model_config.get('azure_endpoint')
    api_version = model_config.get('api_version', '2024-12-01-preview')

    # Provide helpful error messages if required values are missing
    if not api_key:
        raise ValueError(
            f"API key not configured for model '{model_name}'. "
            "Please set the appropriate environment variable (e.g., AZURE_OPENAI_API_KEY) "
            "or check your configs/llm_config.yaml file."
        )

    if not azure_endpoint:
        raise ValueError(
            f"Azure endpoint not configured for model '{model_name}'. "
            "Please set the appropriate environment variable (e.g., AZURE_ENDPOINT) "
            "or check your configs/llm_config.yaml file."
        )

    # Build client parameters from configuration
    client_params = {
        'api_key': api_key,
        'api_version': api_version,
        'azure_endpoint': azure_endpoint,
    }

    # Add optional timeout if specified
    if 'timeout' in model_config:
        client_params['timeout'] = model_config['timeout']

    return AzureOpenAI(**client_params)

def llama_client_init():
    """
    Initialize client for Llama models.
    This is now a convenience wrapper around azure_client_init.
    """
    return azure_client_init(model_name="llama")

def init_client():
    """Initialize default client."""
    return azure_client_init()
    # return openai.OpenAI()

# Removed global clients - now using lazy initialization pattern
_clients = {}

def get_client(model_name: str = None):
    """Get or initialize a client for the specified model."""
    if model_name == "gpt-4.1-nano-2025-04-14":
        key = "nano"
        if key not in _clients:
            _clients[key] = azure_client_init(model_name=model_name)
        return _clients[key]
    elif model_name and "gpt" not in model_name.lower():
        # Llama or other non-GPT models
        key = "llama"
        if key not in _clients:
            _clients[key] = llama_client_init()
        return _clients[key]
    else:
        # Default GPT client
        key = "default"
        if key not in _clients:
            _clients[key] = init_client()
        return _clients[key]

def get_model_response(
    model_name,
    prompt,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1048,
    n=1,
    stream=False,
    method_name: str | None = None,
    recorder: ExperimentRecorder | None = None,
    test_instance: TestInstance | None = None,
):
    """
    OpenAI APIを使用してモデルの応答を取得する
    
    Args:
        model_name: 使用するモデル名（例: "gpt-3.5-turbo"）
        prompt: 入力プロンプト
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        n: 生成する応答の数
    
    Returns:
        応答のテキスト（またはリスト）
    """
    api_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "stream": stream,
    }
    call_role = (
        "answer_sampling"
        if n > 1 or (method_name and method_name.lower() in {"labelprob", "label_prob"})
        else "answer_single"
    )

    try:
        if "gpt" in model_name.lower():
            _client = get_client(model_name)
            response = _client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
            )

            if n == 1:
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API returned None content. This may be due to content filtering or an API error.")
                raw_text = content.strip()
                if recorder and test_instance:
                    recorder.log_api_call(
                        method_name or "unknown",
                        test_instance,
                        call_role,
                        sample_index=0,
                        attempt=1,
                        status="ok",
                        payload={
                            "prompt": prompt,
                            "raw_response_text": raw_text,
                            "api_params": api_params,
                            "provider_request_id": getattr(response, "id", None),
                        },
                    )
                return raw_text
            else:
                results = []
                for idx, choice in enumerate(response.choices):
                    content = choice.message.content
                    if content is None:
                        raise ValueError(f"API returned None content for choice {idx}. This may be due to content filtering or an API error.")
                    raw_text = content.strip()
                    results.append(raw_text)
                    if recorder and test_instance:
                        recorder.log_api_call(
                            method_name or "unknown",
                            test_instance,
                            call_role,
                            sample_index=idx,
                            attempt=1,
                            status="ok",
                            payload={
                                "prompt": prompt,
                                "raw_response_text": raw_text,
                                "api_params": api_params,
                                "provider_request_id": getattr(response, "id", None),
                            },
                        )
                return results

        else:
            stream = True
            _client = get_client(model_name)
            responses = []
            for i in range(n):
                try:
                    response = _client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=1,
                        stream=stream,
                    )
                    buf = []
                    for ch in response:
                        if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                            buf.append(ch.choices[0].delta.content)
                    raw_text = "".join(buf)
                    responses.append(raw_text)
                    if recorder and test_instance:
                        recorder.log_api_call(
                            method_name or "unknown",
                            test_instance,
                            call_role,
                            sample_index=i,
                            attempt=1,
                            status="ok",
                            payload={
                                "prompt": prompt,
                                "raw_response_text": raw_text,
                                "api_params": api_params,
                                "provider_request_id": getattr(response, "id", None),
                            },
                        )
                except Exception as e:
                    if recorder and test_instance:
                        recorder.log_api_call(
                            method_name or "unknown",
                            test_instance,
                            call_role,
                            sample_index=i,
                            attempt=1,
                            status="error",
                            payload={
                                "prompt": prompt,
                                "api_params": api_params,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                    raise
            if stream:
                if len(responses) == 1:
                    return responses[0]
                else:
                    return responses

            if len(responses) == 1:
                content = responses[0].choices[0].message.content
                if content is None:
                    raise ValueError("API returned None content. This may be due to content filtering or an API error.")
                return content.strip()
            else:
                results = []
                for idx, response in enumerate(responses):
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError(f"API returned None content for response {idx}. This may be due to content filtering or an API error.")
                    results.append(content.strip())
                return results

        # print(response)

    except Exception as e:
        if recorder and test_instance:
            recorder.log_api_call(
                method_name or "unknown",
                test_instance,
                call_role,
                sample_index=0,
                attempt=1,
                status="error",
                payload={
                    "prompt": prompt,
                    "api_params": api_params,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise

# def get_model_confidence_with_logprobs(model_name, prompt, temperature=0.0):
#     """
#     """
#     client = openai.OpenAI()
    
#     response = client.chat.completions.create(
#         model=model_name,
#         prompt=prompt,
#         temperature=temperature,
#         max_tokens=256,
#         top_p=1.0
#     )
    
#     answer = response.choices[0].text.strip()
    
#     token_probs = [math.exp(lp) for lp in response.choices[0].logprobs.token_logprobs]
    
#     return answer, confidence

def get_response_with_logprobs(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_logprobs: int = 5,
    return_raw_response: bool = False,
    method_name: str | None = None,
    recorder: ExperimentRecorder | None = None,
    test_instance: TestInstance | None = None,
) -> Tuple[str, Dict]:
    """
    GPT APIを使用して応答を取得し、logprobsも一緒に返す
    
    Args:
        model_name: 使用するモデル名
        prompt: 入力プロンプト
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        logprobs: 返すlogprobsの数
    
    Returns:
        応答のテキストとlogprobsの情報のタプル
    """
    api_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_logprobs": top_logprobs,
        "top_p": 1.0,
    }
    call_role = "answer_sampling" if (method_name and method_name.lower() in {"labelprob", "label_prob"}) else "answer_single"

    try:
        _client = get_client(model_name)

        response = _client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            top_p=1.0
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API returned None content. This may be due to content filtering or an API error.")
        raw_text = content.strip()
        if recorder and test_instance:
            recorder.log_api_call(
                method_name or "unknown",
                test_instance,
                call_role,
                sample_index=0,
                attempt=1,
                status="ok",
                payload={
                    "prompt": prompt,
                    "raw_response_text": raw_text,
                    "api_params": api_params,
                    "provider_request_id": getattr(response, "id", None),
                },
            )

        if return_raw_response:
            return raw_text, response

        logprobs_dict = {"tokens":[], "logprobs":[], "text_offset":[]}
        cumulative_offset = 0
        for logprob in response.choices[0].logprobs.content:
            token_text = logprob.token
            print(f"Token: {token_text}, Probability: {logprob.logprob}")
            logprobs_dict["tokens"].append(token_text)
            logprobs_dict["logprobs"].append(logprob.logprob)
            logprobs_dict["text_offset"].append(cumulative_offset)
            cumulative_offset += len(token_text)

        return raw_text, logprobs_dict

    except Exception as e:
        if recorder and test_instance:
            recorder.log_api_call(
                method_name or "unknown",
                test_instance,
                call_role,
                sample_index=0,
                attempt=1,
                status="error",
                payload={
                    "prompt": prompt,
                    "api_params": api_params,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        raise

# def extract_true_probabilities(response, text_to_find):
#     """
    
#     Args:
        
#     Returns:
#     """
#     result = {}
#     content = response.choices[0].message.content
#     logprobs_data = response.choices[0].logprobs.content
    
#     confidence_matches = list(re.finditer(confidence_pattern, content))
    
#     for match in confidence_matches:
#         truth_value = match.group(2)     # True or False
        
#         true_false_pos = match.start(2)
        
#         closest_token_idx = -1
#         min_distance = float('inf')
        
#         for i, logprob in enumerate(logprobs_data):
#             if hasattr(logprob, 'text_offset'):
#                 distance = abs(logprob.text_offset - true_false_pos)
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_token_idx = i
        
#         if closest_token_idx >= 0:
#             token_data = logprobs_data[closest_token_idx]
            
#             if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
#                 true_prob = 0.0
#                 for candidate in token_data.top_logprobs:
#                     if candidate.token == text_to_find:
#                         break
                
#                 result[confidence_key] = {
#                     'value': truth_value,
#                     f'{text_to_find}_probability': true_prob
#                 }
#             else:
#                 if token_data.token == text_to_find:
#                     result[confidence_key] = {
#                         'value': truth_value,
#                         f'{text_to_find}_probability': math.exp(token_data.logprob)
#                     }
#                 else:
#                     result[confidence_key] = {
#                         'value': truth_value,
#                     }
    
#     return result

def extract_binary_probabilities(response, text_to_find, target_text=("True","False"), debug=False):
    """
    OpenAI API responses から特定のテキスト（例: 'True'）の確率を抽出する

    Args:
        response: OpenAI API レスポンスオブジェクト
        text_to_find: 探したいテキスト（例: 'True'）

    Returns:
        Dict: 各確信度項目の指定テキストの確率を含む辞書
    """
    result = {}
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("API returned None content. This may be due to content filtering or an API error.")
    logprobs_data = response.choices[0].logprobs.content
    
    confidence_pattern = r"(回答確信度|トリプル\d+確信度):\s*(" + target_text[0] + "|" + target_text[1] + ")"
    confidence_matches = list(re.finditer(confidence_pattern, content))
    if debug: print("confidence_matches:", confidence_matches)
    
    for match in confidence_matches:
        confidence_key = match.group(1)
        if debug: print("confidence_key:", confidence_key)
        truth_value = match.group(2)     # True or False
        if debug: print("truth_value:", truth_value)
        
        true_false_pos = match.start(2)
        if debug: print("true_false_pos:", true_false_pos)
        
        closest_token_idx = -1
        min_distance = float('inf')
        
        for i, logprob in enumerate(logprobs_data):
            if hasattr(logprob, 'text_offset'):
                distance = abs(logprob.text_offset - true_false_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_token_idx = i
        
        if debug: print("closest_token_idx:", closest_token_idx)
        if closest_token_idx >= 0:
            token_data = logprobs_data[closest_token_idx]
            if debug: print("token_data:", token_data)
            
            if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                true_prob = 0.0
                for candidate in token_data.top_logprobs:
                    if candidate.token == text_to_find:
                        true_prob = math.exp(candidate.logprob)
                        break
                
                result[confidence_key] = {
                    'value': truth_value,
                    f'{text_to_find}_probability': true_prob
                }
            else:
                if token_data.token == text_to_find:
                    result[confidence_key] = {
                        'value': truth_value,
                        f'{text_to_find}_probability': math.exp(token_data.logprob)
                    }
                else:
                    result[confidence_key] = {
                        'value': truth_value,
                        f'{text_to_find}_probability': 0.0
                    }
    
    return result

from typing import Dict, Any, List, Optional

import math
import re

def safe_get_token_text(token_data):
    """
    トークンデータから安全にテキストを取得
    """
    try:
        if hasattr(token_data, 'bytes') and token_data.bytes is not None:
            if token_data.bytes and len(token_data.bytes) > 0:
                return bytes(token_data.bytes).decode('utf-8')
            else:
                return token_data.token if hasattr(token_data, 'token') else ""
        else:
            return token_data.token if hasattr(token_data, 'token') else ""
    except (UnicodeDecodeError, AttributeError, TypeError):
        try:
            return token_data.token if hasattr(token_data, 'token') else ""
        except:
            return ""


def extract_confidence_probabilities(response):
    """
    正しい確信度抽出：事前確率（top_logprobs）から確信度を計算

    重要：実際に選択されたトークンではなく、選択される前の候補の確率を使用

    Args:
        response: OpenAI API レスポンスオブジェクト

    Returns:
        List[Dict]: 各行の確信度情報
    """
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("API returned None content. This may be due to content filtering or an API error.")
    logprobs_data = response.choices[0].logprobs.content
    
    results = []
    lines = content.splitlines()
    
    print(f"Content lines: {len(lines)}")
    print(f"Logprobs tokens: {len(logprobs_data)}")
    
    truth_line_count = 0
    
    for line_idx, line in enumerate(lines):
        actual_truth_value = None
        if line.strip().endswith('True'):
            actual_truth_value = 'True'
        elif line.strip().endswith('False'):
            actual_truth_value = 'False'
        
        if actual_truth_value:
            print(f"\nProcessing line {line_idx}: '{line}'")
            print(f"  Actual output: {actual_truth_value}")
            
            confidence_data = get_confidence_from_top_logprobs(
                logprobs_data, truth_line_count, actual_truth_value
            )
            
            truth_line_count += 1
            
            if confidence_data:
                results.append({
                    'line_index': line_idx,
                    'line_text': line,
                    'actual_value': actual_truth_value,
                    **confidence_data
                })
                
                print(f"  Confidence results:")
                print(f"    P(True) = {confidence_data['true_probability']:.4f}")
                print(f"    P(False) = {confidence_data['false_probability']:.4f}")
                print(f"    Confidence = {confidence_data['confidence']:.4f}")
                print(f"    Source: {confidence_data['source_description']}")
    
    return results


def get_confidence_from_top_logprobs(logprobs_data, n, actual_truth_value):
    """
    top_logprobsから確信度を取得
    
    戦略：
    1. N番目のTrue/Falseトークンを特定
    2. そのトークンのtop_logprobsから事前確率を取得
    3. True/Falseの相対確率を計算
    """
    
    truth_tokens = []
    
    for i, token_data in enumerate(logprobs_data):
        token_text = safe_get_token_text(token_data)
        
        if is_truth_token(token_text, 'True') or is_truth_token(token_text, 'False'):
            truth_tokens.append((i, token_text, token_data))
    
    print(f"  Found {len(truth_tokens)} truth tokens")
    
    if n >= len(truth_tokens):
        print(f"  Error: Requested token {n} but only found {len(truth_tokens)} tokens")
        return None
    
    token_idx, token_text, token_data = truth_tokens[n]
    print(f"  Using token {n} at index {token_idx}: '{token_text}'")
    
    if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
        return calculate_confidence_from_top_logprobs(
            token_data.top_logprobs, token_idx, actual_truth_value
        )
    else:
        print(f"  No top_logprobs available for token {token_idx}")
        return None


def calculate_confidence_from_top_logprobs(top_logprobs, token_idx, actual_truth_value):
    """
    top_logprobsから確信度を計算
    
    Args:
        top_logprobs: トークンのtop_logprobsリスト
        token_idx: トークンインデックス
        actual_truth_value: 実際の出力値
    
    Returns:
        Dict: 確信度情報
    """
    
    true_logprob = None
    false_logprob = None
    true_token = None
    false_token = None
    
    print(f"    Analyzing top_logprobs for token {token_idx}:")
    
    for i, top in enumerate(top_logprobs):
        top_token_text = safe_get_token_text(top)
        print(f"      {i}: '{top_token_text}' (logprob: {top.logprob:.6f})")
        
        if is_truth_token(top_token_text, 'True') and true_logprob is None:
            true_logprob = top.logprob
            true_token = top_token_text
            print(f"        -> True candidate found")
        
        if is_truth_token(top_token_text, 'False') and false_logprob is None:
            false_logprob = top.logprob
            false_token = top_token_text
            print(f"        -> False candidate found")
    
    if true_logprob is not None and false_logprob is not None:
        exp_true = math.exp(true_logprob)
        exp_false = math.exp(false_logprob)
        total = exp_true + exp_false
        
        true_prob = exp_true / total
        false_prob = exp_false / total
        
        return {
            'true_probability': true_prob,
            'false_probability': false_prob,
            'raw_true_logprob': true_logprob,
            'raw_false_logprob': false_logprob,
            'true_token': true_token,
            'false_token': false_token,
            'calculation_method': 'top_logprobs_normalized',
            'confidence': abs(true_prob - false_prob),
            'source_description': f'From top_logprobs of token {token_idx}'
        }
    
    elif true_logprob is not None:
        true_prob = math.exp(true_logprob)
        false_prob = 1.0 - true_prob
        
        return {
            'true_probability': true_prob,
            'false_probability': false_prob,
            'raw_true_logprob': true_logprob,
            'raw_false_logprob': None,
            'true_token': true_token,
            'false_token': None,
            'calculation_method': 'true_only_from_top_logprobs',
            'confidence': true_prob,
            'source_description': f'True from top_logprobs of token {token_idx}, False estimated',
            'note': 'False probability estimated as 1 - P(True)'
        }
    
    elif false_logprob is not None:
        false_prob = math.exp(false_logprob)
        true_prob = 1.0 - false_prob
        
        return {
            'true_probability': true_prob,
            'false_probability': false_prob,
            'raw_true_logprob': None,
            'raw_false_logprob': false_logprob,
            'true_token': None,
            'false_token': false_token,
            'calculation_method': 'false_only_from_top_logprobs',
            'confidence': false_prob,
            'source_description': f'False from top_logprobs of token {token_idx}, True estimated',
            'note': 'True probability estimated as 1 - P(False)'
        }
    
    else:
        print(f"    Warning: No True/False found in top_logprobs for token {token_idx}")
        return {
            'true_probability': 0.5,
            'false_probability': 0.5,
            'raw_true_logprob': None,
            'raw_false_logprob': None,
            'true_token': None,
            'false_token': None,
            'calculation_method': 'default',
            'confidence': 0.0,
            'source_description': f'No True/False in top_logprobs of token {token_idx}',
            'note': 'Default 50/50 probability assigned'
        }


def is_truth_token(token, target_value):
    """
    トークンが指定された真偽値を表すかどうかを判定
    """
    if not token or not target_value:
        return False
    
    if token == target_value:
        return True
    
    if token.strip() == target_value:
        return True
    
    if token == f" {target_value}":
        return True
    
    if token.rstrip('\n\r ') == target_value:
        return True
    
    cleaned_token = token.strip(' \n\r\t')
    if cleaned_token == target_value:
        return True
    
    if target_value in token:
        return True
    
    return False


def analyze_calibration(results):
    """
    キャリブレーション分析
    
    Args:
        results: extract_confidence_probabilities の結果
    
    Returns:
        Dict: キャリブレーション指標
    """
    if not results:
        return {}
    
    total_lines = len(results)
    correct_predictions = 0
    total_confidence = 0
    
    for result in results:
        predicted_true = result['true_probability'] > 0.5
        actual_true = result['actual_value'] == 'True'
        
        if predicted_true == actual_true:
            correct_predictions += 1
        
        total_confidence += result['confidence']
    
    accuracy = correct_predictions / total_lines
    avg_confidence = total_confidence / total_lines
    
    return {
        'accuracy': accuracy,
        'average_confidence': avg_confidence,
        'total_lines': total_lines,
        'correct_predictions': correct_predictions,
        'calibration_error': abs(accuracy - avg_confidence)
    }


def demo_usage():
    """
    正しい確信度抽出の使用例
    """
    print("=== 正しい確信度抽出の使用例 ===")
    print("results = extract_confidence_probabilities(response)")
    print()
    print("for result in results:")
    print("    print(f'Line {result[\"line_index\"]}: {result[\"line_text\"]}')") 
    print("    print(f'  Actual: {result[\"actual_value\"]}')") 
    print("    print(f'  P(True): {result[\"true_probability\"]:.4f}')") 
    print("    print(f'  P(False): {result[\"false_probability\"]:.4f}')") 
    print("    print(f'  Confidence: {result[\"confidence\"]:.4f}')") 
    print("    print(f'  Source: {result[\"source_description\"]}')") 
    print()
    print("=== キャリブレーション分析 ===")
    print("calibration = analyze_calibration(results)")
    print("print(f'Accuracy: {calibration[\"accuracy\"]:.4f}')")
    print("print(f'Average Confidence: {calibration[\"average_confidence\"]:.4f}')")
    print("print(f'Calibration Error: {calibration[\"calibration_error\"]:.4f}')")


# if __name__ == "__main__":
#     demo_usage()
# def extract_true_probabilities(response, true_text="True", false_text="False"):
#     """

#     Args:

#     Returns:
#     """
#     import re
#     import math
    
#     result = {}
#     content = response.choices[0].message.content
#     logprobs_data = response.choices[0].logprobs.content
    
#     print(f"Content: {content}")
#     print(f"Number of tokens in logprobs: {len(logprobs_data)}")
#     #     print(f"Token {i}: {token_data.token}, logprob: {token_data.logprob}")
#     #     if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
#     #         print(f"  Top logprobs: {[f'{t.token}:{t.logprob}' for t in token_data.top_logprobs[:3]]}")
    
#     reconstructed_text = ""
    
#     for token_data in logprobs_data:
#         token_text = token_data.token
#         start_pos = len(reconstructed_text)
#         reconstructed_text += token_text
#         end_pos = len(reconstructed_text)
#         token_positions.append((start_pos, end_pos))

#     print(f"Reconstructed text length: {len(reconstructed_text)}")
#     print(f"Original content length: {len(content)}")

#     all_matches = list(re.finditer(confidence_pattern, content))
#     # patterns = [
#     # ]
    
#     print(f"Found {len(all_matches)} confidence matches")
    
#     for match in all_matches:
#         truth_value  = match.group(4)
#         truth_start = match.start(4)
#         truth_end  = match.end(4)

#         print(f"\nProcessing: {key} = {truth_value}")
#         print(f"Truth value position: {truth_start}-{truth_end}")
#         print(f"Context: '{content[max(0, truth_start-10):truth_end+10]}'")
#         target_token_idx = None
#         for i, (start_pos, end_pos) in enumerate(token_positions):
#             if start_pos <= truth_start < end_pos or start_pos < truth_end <= end_pos:
#                 target_token_idx = i
#                 break
#             elif truth_start <= start_pos and end_pos <= truth_end:
#                 target_token_idx = i
#                 break
        
#         if target_token_idx is None:
#             min_distance = float('inf')
#             for i, (start_pos, end_pos) in enumerate(token_positions):
#                 distance = min(abs(start_pos - truth_start), abs(end_pos - truth_end))
#                 if distance < min_distance:
#                     min_distance = distance
#                     target_token_idx = i
        
#         print(f"Target token index: {target_token_idx}")
        
#         if target_token_idx is not None:
#             log_p_true = None
#             log_p_false = None
            
#             search_start = max(0, target_token_idx - 3)
#             search_end = min(len(logprobs_data), target_token_idx + 4)
            
#             print(f"Searching tokens {search_start} to {search_end-1}")
            
#             for i in range(search_start, search_end):
#                 token_data = logprobs_data[i]
#                 token_text = token_data.token
                
#                 print(f"  Token {i}: '{token_text}' (logprob: {token_data.logprob:.4f})")
                
#                 if self_contains_truth_value(token_text, true_text):
#                     log_p_true = token_data.logprob
#                     print(f"    -> Found True with logprob {log_p_true}")
                
#                 if self_contains_truth_value(token_text, false_text):
#                     log_p_false = token_data.logprob
#                     print(f"    -> Found False with logprob {log_p_false}")
                
#                 if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
#                     for top in token_data.top_logprobs:
#                         if self_contains_truth_value(top.token, true_text) and log_p_true is None:
#                             log_p_true = top.logprob
#                             print(f"    -> Found True in top_logprobs: '{top.token}' (logprob: {log_p_true})")
                        
#                         if self_contains_truth_value(top.token, false_text) and log_p_false is None:
#                             log_p_false = top.logprob
#                             print(f"    -> Found False in top_logprobs: '{top.token}' (logprob: {log_p_false})")
            
#             result[key] = calculate_probabilities(
#                 log_p_true, log_p_false, truth_value, true_text, false_text
#             )
            
#             print(f"Result for {key}: {result[key]}")
        
#         else:
#             print(f"Could not find token for {key}")
#             result[key] = {
#                 'value': truth_value,
#                 'true_probability': 0.5,
#                 'false_probability': 0.5,
#                 'note': 'Could not locate corresponding token'
#             }
    
#     return result
        
    #     truth_match   = re.search(r'\b(True|False)\b', content[confidence_end:confidence_end+10])
    #     if not truth_match:
    #         continue

    #     actual_truth_value = truth_match.group(1)
    #     truth_start = confidence_end + truth_match.start()
    #     truth_end   = confidence_end + truth_match.end()

        
    #     print(f"Processing: {confidence_key} with value {actual_truth_value} (match end: {match.end()})")
        
    #     # truth_match = re.search(r'\s*(True|False)', content[confidence_end:confidence_end+10])
    #     # if not truth_match:
    #     #     print(f"Could not find True/False after {confidence_key}")
    #     #     result[confidence_key] = {
    #     #         'value': actual_truth_value,
    #     #         'note': 'Could not locate True/False position'
    #     #     }
    #     #     continue
        
    #     # truth_start = confidence_end + truth_match.start()
    #     # truth_end = confidence_end + truth_match.end()
    #     # actual_truth_value = truth_match.group(1)
        
    #     print(f"Found {actual_truth_value} at position {truth_start}-{truth_end}")
        
    #     closest_token_idx = None
    #     min_distance = float('inf')
        
    #     for i, boundary in enumerate(token_boundaries):
    #         if abs(boundary - truth_start) < min_distance:
    #             min_distance = abs(boundary - truth_start)
    #             closest_token_idx = i
        
    #     log_p_true = None
    #     log_p_false = None
        
    #     print(f"Searching around token index {closest_token_idx}")
        
    #     for i in range(max(0, closest_token_idx - search_range), min(len(logprobs_data), closest_token_idx + search_range)):
    #         token_data = logprobs_data[i]
    #         print(f"Checking token {i}: '{token_data.token}'")
            
    #         if token_data.token == true_text or token_data.token == f" {true_text}" or token_data.token.endswith(true_text):
    #             log_p_true = token_data.logprob
    #             print(f"Found True token at {i} with logprob {log_p_true}")
            
    #         elif token_data.token == false_text or token_data.token == f" {false_text}" or token_data.token.endswith(false_text):
    #             log_p_false = token_data.logprob
    #             print(f"Found False token at {i} with logprob {log_p_false}")
            
    #         if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
    #             for top in token_data.top_logprobs:
    #                 if (top.token == true_text or top.token == f" {true_text}" or 
    #                     top.token.endswith(true_text)) and log_p_true is None:
    #                     log_p_true = top.logprob
    #                     print(f"Found True in top_logprobs at {i} with logprob {log_p_true}")
                    
    #                 if (top.token == false_text or top.token == f" {false_text}" or 
    #                     top.token.endswith(false_text)) and log_p_false is None:
    #                     log_p_false = top.logprob
    #                     print(f"Found False in top_logprobs at {i} with logprob {log_p_false}")
        
    #     if log_p_true is None or log_p_false is None:
    #         print("Extended search through all tokens")
    #         for i, token_data in enumerate(logprobs_data):
    #             token_text = token_data.token
    #             if true_text in token_text and log_p_true is None:
    #                 log_p_true = token_data.logprob
    #                 print(f"Extended search: Found True in token at {i}: '{token_text}' with logprob {log_p_true}")
                
    #             if false_text in token_text and log_p_false is None:
    #                 log_p_false = token_data.logprob
    #                 print(f"Extended search: Found False in token at {i}: '{token_text}' with logprob {log_p_false}")
                
    #             if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
    #                 for top in token_data.top_logprobs:
    #                     if true_text in top.token and log_p_true is None:
    #                         log_p_true = top.logprob
    #                         print(f"Extended search: Found True in top_logprobs at {i}: '{top.token}' with logprob {log_p_true}")
                        
    #                     if false_text in top.token and log_p_false is None:
    #                         log_p_false = top.logprob
    #                         print(f"Extended search: Found False in top_logprobs at {i}: '{top.token}' with logprob {log_p_false}")
        
    #     if log_p_true is None or log_p_false is None:
    #         context_start = max(0, closest_token_idx - 10)
    #         context_end = min(len(logprobs_data), closest_token_idx + 10)
            
    #         print(f"Full context around confidence pattern (tokens {context_start}-{context_end}):")
    #         for i in range(context_start, context_end):
    #             token_data = logprobs_data[i]
    #             print(f"Token {i}: '{token_data.token}', logprob: {token_data.logprob}")
    #             if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
    #                 print(f"  Top logprobs: {[f'{t.token}:{t.logprob}' for t in token_data.top_logprobs[:5]]}")
        
    #     if log_p_true is not None and log_p_false is not None:
    #         p_true = math.exp(log_p_true)
    #         p_false = math.exp(log_p_false)
    #         prob_true = p_true / (p_true + p_false)
            
    #         result[confidence_key] = {
    #             'value': truth_value,
    #             'true_probability': prob_true,
    #             'false_probability': 1.0 - prob_true,
    #             'raw_true_logprob': log_p_true,
    #             'raw_false_logprob': log_p_false
    #         }
            
    #         print(f"Normalized probability for True: {prob_true:.4f}, False: {1.0 - prob_true:.4f}")
        
    #     elif log_p_true is not None:
    #         p_true = math.exp(log_p_true)
    #         prob_true = p_true / (p_true + p_false)
            
    #         result[confidence_key] = {
    #             'value': truth_value,
    #             'true_probability': prob_true,
    #             'false_probability': 1.0 - prob_true,
    #             'raw_true_logprob': log_p_true,
    #             'note': 'Only True probability found, False assumed to be near zero'
    #         }
    #         print(f"Only True probability found. Assumed True: {prob_true:.4f}, False: {1.0 - prob_true:.4f}")
        
    #     elif log_p_false is not None:
    #         p_false = math.exp(log_p_false)
    #         prob_true = p_true / (p_true + p_false)
            
    #         result[confidence_key] = {
    #             'value': truth_value,
    #             'true_probability': prob_true,
    #             'false_probability': 1.0 - prob_true,
    #             'raw_false_logprob': log_p_false,
    #             'note': 'Only False probability found, True assumed to be near zero'
    #         }
    #         print(f"Only False probability found. Assumed True: {prob_true:.4f}, False: {1.0 - prob_true:.4f}")
        
    #     else:
    #         print("No probabilities found. Searching through all tokens for any True/False:")
    #         found_true = []
    #         found_false = []
            
    #         for i, token_data in enumerate(logprobs_data):
    #             if true_text in token_data.token:
    #                 found_true.append((i, token_data.token, token_data.logprob))
                
    #             if false_text in token_data.token:
    #                 found_false.append((i, token_data.token, token_data.logprob))
                
    #             if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
    #                 for top in token_data.top_logprobs:
    #                     if true_text in top.token:
    #                         found_true.append((i, f"top:{top.token}", top.logprob))
                        
    #                     if false_text in top.token:
    #                         found_false.append((i, f"top:{top.token}", top.logprob))
            
    #         print(f"Found True in tokens: {found_true}")
    #         print(f"Found False in tokens: {found_false}")
            
    #         actual_truth_value = content[truth_start:truth_end].strip()
    #         print(f"Actual truth value in content: '{actual_truth_value}', Expected: '{truth_value}'")
            
    #         if found_true and truth_value == "True":
    #             log_p_true = found_true[0][2]
    #             p_true = math.exp(log_p_true)
    #             p_false = 1e-10
    #             prob_true = p_true / (p_true + p_false)
                
    #             result[confidence_key] = {
    #                 'value': truth_value,
    #                 'true_probability': prob_true,
    #                 'false_probability': 1.0 - prob_true,
    #                 'raw_true_logprob': log_p_true,
    #                 'note': 'Found True token elsewhere, False assumed to be near zero'
    #             }
    #             print(f"Using True from elsewhere. Assumed True: {prob_true:.4f}, False: {1.0 - prob_true:.4f}")
            
    #         elif found_false and truth_value == "False":
    #             log_p_false = found_false[0][2]
    #             p_false = math.exp(log_p_false)
    #             p_true = 1e-10
    #             prob_true = p_true / (p_true + p_false)
                
    #             result[confidence_key] = {
    #                 'value': truth_value,
    #                 'true_probability': prob_true,
    #                 'false_probability': 1.0 - prob_true,
    #                 'raw_false_logprob': log_p_false,
    #                 'note': 'Found False token elsewhere, True assumed to be near zero'
    #             }
    #             print(f"Using False from elsewhere. Assumed True: {prob_true:.4f}, False: {1.0 - prob_true:.4f}")
            
    #         else:
    #             result[confidence_key] = {
    #                 'value': truth_value,
    #                 'true_probability': 0.5,
    #                 'false_probability': 0.5,
    #                 'note': 'No probabilities found for True or False, using default 50/50'
    #             }
    #             print(f"No probabilities found for {confidence_key}, using default 50/50")
    
    # return result

# result = extract_true_probabilities(response, "True")
# print(result)

def run_two_step_evaluation(model_name, prompt1s, prompt2s, extract_answer_func=None, extract_prob_func=None, temperature=0.0, prompt2_param=None, logprobs=False, top_logprobs=5, max_tokens=512,debug=False):
    """
    2ステップでの評価を実行する関数
    """
    _client = get_client(model_name)

    messages = [
        {"role": "system", "content": "あなたは質問に正確に答えるアシスタントです。"},
        {"role": "user", "content": prompt1s}
    ]

    response1 = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    step1_response = response1.choices[0].message.content
    if step1_response is None:
        raise ValueError("API returned None content in step 1. This may be due to content filtering or an API error.")
    answers = parse_output(step1_response)

    # answers = extract_answer_func(step1_response)

    if debug:
        print('run_two_step_evaluation: prompt1s', prompt1s)
        print('run_two_step_evaluation: step1_response', step1_response)
    
    messages.append({"role": "assistant", "content": step1_response})
    
    if prompt2_param:
        for k, v in prompt2_param.items():
            if v == "dummy_answer":
                # v =     answers = 
                v = answers.answers[0]

            prompt2s = prompt2s.replace(k, v)
    messages.append({"role": "user", "content": prompt2s})
    
    # response2 = client.chat.completions.create(
    #     model=model_name,
    #     messages=messages,
    #     temperature=temperature
    # )
    
    # step2_response = response2.choices[0].message.content
    # probabilities = extract_prob_func(step2_response)
    if logprobs:
        response2 = _client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            top_p=1.0
        )
    else:
        response2 = _client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    step2_response = response2.choices[0].message.content
    if step2_response is None:
        raise ValueError("API returned None content in step 2. This may be due to content filtering or an API error.")
    # probabilities = extract_prob_func(step2_response)
    probabilities = parse_output(step2_response)

    if debug:
        print('run_two_step_evaluation: prompt2s', prompt2s)
        print('run_two_step_evaluation: step2_response', step2_response)

    if logprobs:
        return answers, probabilities, response2

    return answers, probabilities

def check_answer_equivalence(answer1: str, answer2: str, question: str) -> bool:
    """
    2つの回答が意味的に同等かを評価する
    """
    _client = get_client("gpt-4o-2024-11-20")

    prompt = f"""Are the following two answers to this question semantically equivalent?
    
Question: {question}
Answer 1: {answer1}
Answer 2: {answer2}

Reply with only "Yes" or "No".
"""

    response = _client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("API returned None content. This may be due to content filtering or an API error.")
    result = content.strip().lower()
    return "yes" in result



