from finegrained_conf.llm.openai_client import *
# from prompt_templates import *
# from prompt_templates_triple import *
from finegrained_conf.prompts.answer_prompts import *
from finegrained_conf.prompts.evidence_prompts import *
import json

def normalize(ans):
    ans = ans.strip().lower()
    token = re.split(r'[、,。. ]', ans)[0]
    return {'yes':'YES', 'はい':'YES',
            'no':'NO', 'いいえ':'NO'}.get(token, token)
def check_answer_correctness(prediction, gold_answer, question, model="gpt-4o-2024-11-20", language="ja", debug=False):
    if debug:
        print(f"check_answer_correctness: question={question}, gold_answer={gold_answer}, prediction={prediction}, model={model}")
    if prediction.lower() == gold_answer.lower():
        return True

    if normalize(prediction) in ['YES', 'NO'] and normalize(gold_answer) in ['YES', 'NO']:
        if normalize(prediction)==normalize(gold_answer):
            return True
        else:
            return False
    
    if language=="ja":
        evaluation_prompt_template=EVALUATION_PROMPT_JP
    elif language=="en":
        evaluation_prompt_template=EVALUATION_PROMPT_EN
    prompt = evaluation_prompt_template.replace("${THE_QUESTION}", question) \
                                       .replace("${GOLD_ANSWER}", gold_answer) \
                                       .replace("${PRED_ANSWER}", prediction)
    
    response = get_model_response(model, prompt, max_tokens=2048)
    print(f"check_answer_correctness: prompt={prompt}")
    print(f"check_answer_correctness: response={response}")
    
    if "Yes" in response or "はい" in response:
        return True
    
    return False

evaluate_triples_schema_en = {
    "name": "evaluate_triples",
    "description": """Match predicted triples and gold triples one-to-one using the Hungarian algorithm, and return a JSON array where each pair receives a score (1.0 or 0.0).

Triples that should score 1.0
- Exact match
- Synonyms / spelling variants / abbreviations
- Information equivalence: contains the information needed to answer the question
  - Differences in numeric units or granularity (as long as the required granularity for the question is satisfied)
  - Traditional age vs. full age; rounding to the same bucket where the answer does not change
- Subject↔Object swap for symmetric relations (e.g., spouse, partnership)
- Presuppositions: question premises, temporal prerequisites, or confirmation of known facts, allowed even if absent from gold
- Composite chain: multiple preds together can exactly derive the gold value (all preds required)
- Inference elements:
  - Indirect but necessary relational information for reasoning
  - Boundary values of time ranges (e.g., start/end year of the Shōwa era)
  - Components of composite conditions (each element of an AND condition)
  - Details required for comparisons (e.g., month information when the year is the same)

Triples that should score 0.0
1) Missing or misidentified core information:
   - Ignoring additional requirements such as “both” or “the other”
   - Providing only part of the required complete information (unless it functions as an inference element)
2) Factually incorrect information (excluding conventional differences)
3) Improper format: subject/object are not entities/values; boolean values, bare comparison expressions, or full sentences
4) Vague expressions (“around,” “many,” etc.)
5) Missing elements in a composite chain
6) Irrelevant to the question or the gold

Principles for judgment:
- Prioritize contribution to the reasoning process
- Emphasize derivability within composite chains
- Partial information still scores 1.0 if it contributes to reasoning
- When in doubt, judge by the degree of contribution to reasoning
""",
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["pred", "gold"],
                            "description": "Indicates whether this item is a predicted triple or a gold triple."
                        },
                        "index": {
                            "type": "integer",
                            "description": "1-based index in the original list."
                        },
                        "triple": {
                            "type": "string",
                            "description": "Triple in the form (subject entity, relation, object entity/value)."
                        },
                        "matched_index": {
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "null"}
                            ],
                            "description": "Index of the matched counterpart triple; null if unmatched."
                        },
                        "score": {
                            "type": "number",
                            "enum": [0.0, 1.0],
                            "description": "Evaluation score (1.0 or 0.0)."
                        }
                    },
                    "required": ["type", "index", "triple", "matched_index", "score"]
                }
            }
        },
        "required": ["results"]
    }
}

evaluate_triples_schema_jp = {
    "name": "evaluate_triples",
    "description": """予測トリプルと正解トリプルをハンガリアン法で1対1マッチングし、各ペアにスコア（1.0 or 0.0）を付与してJSON配列を返す。
▼1.0 になる pred
- 完全一致
- 同義語／表記ゆれ／略称
- 情報等価：質問回答に必要な情報を含む
  - 数値の単位・粒度差（質問に必要な粒度を満たす場合）
  - 数え年⇔満年齢、同じ区分に丸めても答えが変わらない場合
- 対称関係の主語⇔目的語入れ替え（配偶者・提携 等）
- 前提情報：質問の前提条件・時系列的前提・既知情報の確認、goldに無い場合も許容
- 複合チェイン：複数predでgold値を正確に導出可能（全pred必要）
- 推論要素：
  - 間接的だが推論に必要な関係情報
  - 時間範囲の境界値（昭和の開始/終了年等）
  - 複合条件の部分要素（AND条件の各要素）
  - 比較に必要な詳細（同年なら月情報等）

▼0.0 になる pred
1) 核心情報の欠落・誤認：
   - 「両方」・「もう一つの」等の追加要求無視
   - 要求された完全情報の一部のみ（※推論要素として機能する場合は除く）
2) 事実と異なる情報（※慣習的差異は除く）
3) 不適切な形式：主語・目的語がエンティティ／値でなく、Bool値、単純比較式、文章
4) 曖昧表現（「頃」「多数」等）
5) 複合チェインの要素不足
6) 質問・goldと無関係

**判定原則**：
- 推論への寄与を最優先評価
- 複合チェインでの導出可能性を重視
- 部分情報でも推論に寄与すれば1.0
- 迷ったら推論への寄与度で判定
""",
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["pred", "gold"],
                            "description": "予測か正解かを示す"
                        },
                        "index": {
                            "type": "integer",
                            "description": "元リストにおける1始まりのインデックス"
                        },
                        "triple": {
                            "type": "string",
                            "description": "（主語エンティティ, 関係, 目的語エンティティ）の形式のトリプル"
                        },
                        "matched_index": {
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "null"}
                            ],
                            "description": "対応する相手トリプルのインデックス、未対応時はnull"
                        },
                        "score": {
                            "type": "number",
                            "enum": [0.0, 1.0],
                            "description": "評価スコア（1.0または0.0）"
                        }
                    },
                    "required": ["type", "index", "triple", "matched_index", "score"]
                }
            }
        },
        "required": ["results"]
    }
}

def check_triples_correctness(pred_triples, gold_triples, question, model="gpt-4o-2024-11-20", evaluation_prompt_template=TRIPLE_EVALUATION_PROMPT_JP, language="ja", debug=False):

    if debug:
        print(f"check_triples_correctness: question={question}, gold_triples={gold_triples}, pred_triples={pred_triples}, model={model}")

    if not None in pred_triples:
        if ';'.join(pred_triples).replace('（','(').replace('）',')').lower() == gold_triples.replace('（','(').replace('）',')').lower():
            return [1.0] * len(pred_triples), [1.0] * len(gold_triples)
    if all(x is None for x in pred_triples):
        return [None] * len(pred_triples), [None] * len(gold_triples)

    pred_triples_str = []
    if language=="ja":
        for i, pred_t in enumerate(pred_triples):
            pred_triples_str.append(f"予測トリプル{i+1}: " + pred_t)
        pred_triples_strs = "\n".join(pred_triples_str)
        eval_triples_schema = evaluate_triples_schema_jp
        system_content = "あなたはトリプル評価エンジンです。必ず evaluate_triples 関数を呼び出し、指定されたスコア基準に従って評価してください。出力順序は pred の1,2,... → gold の1,2,... の順です。"
        user_content = (
            "以下を評価してください。"
            f"質問: {question}\n"
            f"正解トリプル: {gold_triples}\n"
            f"予測トリプル: {pred_triples_strs}"
        )
    elif language=="en":
        for i, pred_t in enumerate(pred_triples):
            pred_triples_str.append(f"Predicted triple {i+1}: " + pred_t)
        pred_triples_strs = "\n".join(pred_triples_str)
        eval_triples_schema = evaluate_triples_schema_en
        system_content = "You are a triple evaluation engine. You must call the evaluate_triples function and evaluate according to the specified scoring criteria. The output order must be pred 1,2,... → gold 1,2,... in that order."
        user_content = (
            "Please evaluate the following."
            f"Question: {question}\n"
            f"Gold triples: {gold_triples}\n"
            f"Predicted triples: {pred_triples_strs}"
        )

    # Retry logic for handling JSON decode errors and API failures
    max_retries = 3
    _client = get_client(model)

    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model,
                max_tokens=8192,  # Increased from 4096 to handle more triples
                temperature=0.0,
                n=1,
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                functions=[eval_triples_schema],
                function_call={"name": "evaluate_triples"}
            )

            if debug:
                print("check_triples_correctness: response=\n"+json.dumps(response.to_dict(), ensure_ascii=False, indent=2))

            func_args = response.choices[0].message.function_call.arguments

            data = json.loads(func_args)

            # If we successfully parsed JSON, break out of retry loop
            break

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {e}")
            if debug and 'func_args' in locals():
                print(f"Failed to parse JSON: {func_args[:500]}...")  # Print first 500 chars

            if attempt == max_retries - 1:
                # On final attempt, raise the error
                raise
            else:
                # Wait a bit before retrying
                import time
                time.sleep(1)
                continue

        except Exception as e:
            print(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            else:
                import time
                time.sleep(2)
                continue

    correctness_list = []
    gold_correctness_list = []
    pred_triples = []
    gold_triples = []

    # print(data)

    for entry in data['results']:
        # print(entry)
        if entry.get('type', '') == "pred":
            correctness_list.append(entry.get('score', ''))
            pred_triples.append(entry.get('triple', ''))
        elif entry.get('type', '') == "gold":
            gold_correctness_list.append(entry.get('score', ''))
            gold_triples.append(entry.get('triple', ''))

    # print(data)
    # print(f"check_triples_correctness: pred_triples={pred_triples}")
    # print(f"check_triples_correctness: correctness_list={correctness_list}")
    # print(f"check_triples_correctness: gold_triples={gold_triples}")
    # print(f"check_triples_correctness: gold_correctness_list={gold_correctness_list}")
    

    return correctness_list, gold_correctness_list