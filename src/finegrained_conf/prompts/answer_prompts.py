LINGUISTIC_EXPRESSIONS = "Almost Certain, Highly Likely, Very Good Chance, Probably, Likely, Better than Even, About Even, Probably Not, Unlikely, Little Chance, Chances are Slight, Highly Unlikely, Almost No Chance"

LINGUISTIC_TO_PROB = {
    "Almost Certain": 0.95,
    "Highly Likely": 0.85,
    "Very Good Chance": 0.75,
    "Probably": 0.65,
    "Likely": 0.60,
    "Better than Even": 0.55,
    "About Even": 0.50,
    "Probably Not": 0.35,
    "Unlikely": 0.25,
    "Little Chance": 0.20,
    "Chances are Slight": 0.15,
    "Highly Unlikely": 0.10,
    "Almost No Chance": 0.05
}


LINGUISTIC_EXPRESSIONS_JP = "ほぼ確実、非常に可能性が高い、かなり可能性がある、おそらく、可能性が高い、五分五分より高い、五分五分、おそらく違う、可能性が低い、あまり可能性がない、わずかな可能性、非常に可能性が低い、ほぼ可能性がない"

LINGUISTIC_TO_PROB_MAPPING_JP = {
    "ほぼ確実": 0.95,
    "非常に可能性が高い": 0.85,
    "かなり可能性がある": 0.75,
    "おそらく": 0.65,
    "可能性が高い": 0.60,
    "五分五分より高い": 0.55,
    "五分五分": 0.50,
    "おそらく違う": 0.35,
    "可能性が低い": 0.25,
    "あまり可能性がない": 0.20,
    "わずかな可能性": 0.15,
    "非常に可能性が低い": 0.10,
    "ほぼ可能性がない": 0.05
}

LINGUISTIC_EXPRESSIONS_7JP = "ほぼ確実、かなり高い、やや高い、どちらともいえない、やや低い、非常に低い、ほぼない"

LINGUISTIC_TO_PROB_7JP = {
    "ほぼ確実":       0.95,
    "かなり高い":     0.75,
    "やや高い":       0.60,
    "どちらともいえない": 0.50,
    "やや低い":       0.40,
    "非常に低い":     0.15,
    "ほぼない":       0.05,
}

PROB_TO_LINGUISTIC_MAPPING_JP = {
    0.95: "ほぼ確実",
    0.85: "非常に可能性が高い",
    0.75: "かなり可能性がある",
    0.65: "おそらく",
    0.60: "可能性が高い",
    0.55: "五分五分より高い",
    0.50: "五分五分",
    0.35: "おそらく違う",
    0.25: "可能性が低い",
    0.20: "あまり可能性がない",
    0.15: "わずかな可能性",
    0.10: "非常に可能性が低い",
    0.05: "ほぼ可能性がない"
}


PROMPT_TEMPLATES_EN = {
    "label_prob": """Provide the single best answer to the question below. Output the answer only; do not include any other words or explanations.

Question: {THE_QUESTION}

Answer in this format:
Answer: <the most likely answer, as short as possible; not a full sentence—answer only!>
""",

    "verb_1s_top_1": """Provide the single best answer to the question below and the probability (0.0 to 1.0) that your answer is correct. Output the answer and the probability only; do not include any other words or explanations.

Question: {THE_QUESTION}

Answer in this format:
Answer: <the most likely answer, as short as possible; not a full sentence—answer only!>
Probability: <the probability that your answer is correct (between 0.0 and 1.0), with no extra comments—number only!>

""",

    "ling_1s_human": """Provide the single best answer to the question below, and express how likely your answer is to be correct using one of the following expressions: {EXPRESSION_LIST}
Question: {THE_QUESTION}

Answer in this format:
Answer: <the most likely answer, as short as possible; not a full sentence—answer only!>
Answer confidence: <one short phrase from the list above, with no extra comments—phrase only!>

""",

    # One-step prompt that asks the model to use Chain-of-Thought to generate the answer
    "verb_1s_cot": """Show the step-by-step process you use to think through the following question. Then provide your answer and the probability (0.0 to 1.0) that your answer is correct.

Question: {THE_QUESTION}

Respond in this format:
Thought: <a brief, one-sentence summary of your reasoning>
Answer: <the most likely answer, as short as possible; not a full sentence—answer only!>
Probability: <the probability that your answer is correct (between 0.0 and 1.0), with no extra comments—number only!>
""",
}

# PROMPT_TEMPLATES = {
#     "label_prob": """Provide your best guess for the following question. Give ONLY the guess, no
# other words or explanation.

# For example:

# Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

# The question is: ${THE_QUESTION}""",

#     "is_true_prob": """Question: ${THE_QUESTION}
# Proposed Answer: ${ANSWER}
# Is the proposed answer:
# \t(A) True or
# \t(B) False?
# The proposed answer is:""",


#     "verb_1s_top_1": """Provide your best guess and the probability that it is correct (0.0 to 1.0) for
# the following question. Give ONLY the guess and probability, no other words or
# explanation. For example:

# Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!>
# Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>

# The question is: ${THE_QUESTION}""",

#     "verb_1s_top_2": """Provide your 2 best guesses and the probability that each is correct (0.0 to
# 1.0) for the following question. Give ONLY the guesses and probabilities, no other
# words or explanation. For example:

# G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>
# P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>
# G2: <second most likely guess, as short as possible; not a complete sentence, just the guess!>
# P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>

# The question is: ${THE_QUESTION}""",

#     "verb_1s_top_4": """Provide your 4 best guesses and the probability that each is correct (0.0 to
# 1.0) for the following question. Give ONLY the guesses and probabilities, no other
# words or explanation. For example:

# G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>
# P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>
# G2: <second most likely guess, as short as possible; not a complete sentence, just the guess!>
# P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>
# G3: <third most likely guess, as short as possible; not a complete sentence, just the guess!>
# P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>
# G4: <fourth most likely guess, as short as possible; not a complete sentence, just the guess!>
# P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>

# The question is: ${THE_QUESTION}""",


#     "verb_1s_cot": """Provide your best guess for the following question. Before giving your answer,
# provide a step-by-step explanation of your thought process. Then give the guess with
# the probability that it is correct (0.0 to 1.0). Give ONLY the explanation, guess,
# and probability, no other words.

# For example:

# Explanation: <one sentence step-by-step explanation of your thought process>
# Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!>
# Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>

# The question is: ${THE_QUESTION}""",


#     "verb_2s_cot": """Provide your best guess for the following question. Before giving your answer,
# provide a step-by-step explanation of your thought process. Then on a new line
# give the guess with no other words or explanation.

# For example:

# Explanation: <one sentence step-by-step explanation of your thought process>
# Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

# The question is: ${THE_QUESTION}""",

#     "verb_2s_cot_prob": """Provide the probability that your guess is correct. Give ONLY the probability, no
# other words or explanation.

# For example:

# Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
# """,

#     "verb_2s_top_1": """Provide your best guess for the following question. Give ONLY the guess, no
# other words or explanation.

# For example:

# Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

# The question is: ${THE_QUESTION}""",

#     "verb_2s_top_1_prob": """Provide the probability that your guess is correct. Give ONLY the probability, no
# other words or explanation.

# For example:

# Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
# """,

#     "verb_2s_top_k": """Provide your ${k} best guesses for the following question. Give ONLY the guesses,
# no other words or explanation. For example:

# G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>
# G2: <second most likely guess, as short as possible; not a complete sentence, just the guess!>
# ...
# G${k}: <${k}-th most likely guess, as short as possible; not a complete sentence, just the guess!>

# The question is: ${THE_QUESTION}""",

#     "verb_2s_top_k_prob": """Provide the probability that each of your guesses is correct. Give ONLY
# the probabilities, no other words or explanation.

# For example:

# P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>
# ...
# P${k}: <the probability between 0.0 and 1.0 that G${k} is correct, without any extra commentary whatsoever; just the probability!>
# """,


#     "ling_1s_human": """Provide your best guess for the following question, and describe how likely it is
# that your guess is correct as one of the following expressions: ${EXPRESSION_LIST}.

# Question: {THE_QUESTION}

# Provide your response in the following format:
# Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!> confidence_expression
# """,
# }



# EVALUATION_PROMPT = """Are the following two answers to my
# question Q semantically equivalent?

# Q: ${THE_QUESTION}
# A1: ${GOLD_ANSWER}
# A2: ${PRED_ANSWER}

# Please answer with a single word, either "Yes." or "No.", and
# explain your reasoning."""

EVALUATION_PROMPT_EN = """Are the following two answers semantically equivalent for the given question?
Evaluation points:
- Allow differences in wording or granularity if they are appropriate to the question; however, answers at a higher-level (more general) concept than what is asked are not acceptable.
- For dates: if the question requires evidence at year–month–day granularity, an exact match is required; otherwise, year-only or year–month is acceptable if sufficient to answer the question.
- For numbers: allow rounding.

Question: ${THE_QUESTION}
Answer 1: ${GOLD_ANSWER}
Answer 2: ${PRED_ANSWER}

Respond with only the single word "Yes" or "No", and then explain your reasoning."""


PROMPT_TEMPLATES_JP = {
    "label_prob": """次の質問に対する最も良い回答を提供してください。回答「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
""",
    "label_prob_cot": """次の質問に対する答えを考えるプロセスを段階的に示してください。その後、最終的な回答を提供してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
思考: <あなたの思考過程を説明>
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
""",
    "verb_1s_top_1": """次の質問に対する最も良い回答と、その回答が正しい確率（0.0から1.0）を提供してください。回答と確率「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>

""",

    "verb_1s_top_k": """次の質問に対する{k}つの最も良い回答候補と、それぞれの回答が正しい確率（0.0から1.0）を提供してください。回答と確率「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答1: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率1: <回答1が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
...
回答{k}: <{k}番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率{k}: <回答{k}が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
""",

    "verb_1s_top_2": """次の質問に対する2つの最も良い回答候補と、それぞれの回答が正しい確率（0.0から1.0）を提供してください。回答と確率「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答1: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率1: <回答1が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
回答2: <2番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率2: <回答2が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
""",

    "verb_1s_top_4": """次の質問に対する4つの最も良い回答候補と、それぞれの回答が正しい確率（0.0から1.0）を提供してください。回答と確率「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答1: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率1: <回答1が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
回答2: <2番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率2: <回答2が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
回答3: <3番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率3: <回答3が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
回答4: <4番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率4: <回答4が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
""",

    "verb_2s_cot": """次の質問に対する最も良い回答を提供してください。回答の前に、あなたの思考過程を段階的に説明してください。その後、新しい行に回答のみを書いてください、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
説明: <あなたの思考過程を一文で段階的に説明>
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
""",

    "verb_2s_cot_prob": """あなたの回答が正しい確率を提供してください。確率「のみ」を出力し、他の言葉や説明は含めないでください。
以下の形式で回答してください：
確率: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>""",

    "verb_2s_top_1": """次の質問に対する最も良い回答を提供してください。回答「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
""",

    "verb_2s_top_1_prob": """あなたの回答が正しい確率を提供してください。確率「のみ」を出力し、他の言葉や説明は含めないでください。
以下の形式で回答してください：
確率: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>""",

    "verb_2s_top_k": """次の質問に対する{k}つの最も良い回答候補を提供してください。回答「のみ」を出力し、他の言葉や説明は含めないでください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答1: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
...
回答{k}: <{k}番目に可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
""",

    "verb_2s_top_k_prob": """あなたの各回答候補が正しい確率を提供してください。確率「のみ」を出力し、他の言葉や説明は含めないでください。

以下の形式で回答してください：
確率1: <回答1が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
...
確率{k}: <回答{k}が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>""",

    "ling_1s_human": """次の質問に対する最も良い回答を提供し、以下の表現のいずれかを使ってあなたの回答が正しい確率を表現してください: {EXPRESSION_LIST}
質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>

""",
    "ling_2s_human": """あなたが提供した回答について、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください：{EXPRESSION_LIST}
    
以下の形式で回答してください：
回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
...
""",
    "is_true_prob": """あなたが提供した回答について、その正確さに対する確信度を評価してください。
質問: ${THE_QUESTION}
提案された回答: ${ANSWER}
以下の形式で回答してください：
回答確信度: <(A) 正しい または(B) 間違っている、追加のコメントは一切なく；(A) または(B)だけを！>""",
    "is_true_logprob": """あなたが提供した回答の正確さをTrue/Falseで評価してください。
以下の形式で回答してください：
回答確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
...""",
    "is_true_logprob_binary": """あなたが提供した回答の正確さを0/1で評価してください。
以下の形式で回答してください：
回答確信度: <0または1、追加のコメントは一切なく；0または1だけを！>
...""",
    "verb_1s_cot": """次の質問に対する答えを考えるプロセスを段階的に示してください。その後、回答と、その回答が正しい確率（0.0から1.0）を提供してください。

質問: ${THE_QUESTION}

以下の形式で回答してください：
思考: <あなたの思考過程を一文で端的に説明>
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
確率: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
""",
    "verb_1s_cot_is_true": """次の質問に対する答えを考えるプロセスを段階的に示してください。その後、回答とその回答の正確さをTrue/Falseで評価してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
思考: <あなたの思考過程を一文で段階的に説明>
回答: <最も可能性の高い回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
""",

}





EVALUATION_PROMPT_JP = """次の質問に対する2つの回答は意味的に同等でしょうか？
評価ポイント:
・ 表記が異なっても質問に適切なら粒度の違いを許容するが、問われているより上位概念での回答は不可
・ 日付は年月日が根拠として必要な質問では完全一致が必要であるが、質問への回答に十分であれば年、年月のみでも許容
・ 数値は四捨五入を許容

質問: ${THE_QUESTION}
回答1: ${GOLD_ANSWER}
回答2: ${PRED_ANSWER}

「はい」または「いいえ」の単語だけで回答し、その後に理由を説明してください。"""

# fix4eval





# """


def get_closest_linguistic_expression(probability, mapping_dict=PROB_TO_LINGUISTIC_MAPPING_JP):
    """
    確率値に最も近い言語表現を返す関数
    
    Args:
        probability: 0から1の間の確率値
        
    Returns:
        確率値に最も近い言語表現
    """
    probability = min(max(probability, 0.0), 1.0)
    
    probs = list(mapping_dict.keys())
    
    closest_prob = min(probs, key=lambda x: abs(x - probability))
    
    return mapping_dict[closest_prob]

