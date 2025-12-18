# Prompt templates for triple-based confidence evaluation

TRIPLE_PROMPT_TEMPLATES_EN = {
    "triple_label_prob": """Provide the answer and the supporting triples that lead to your conclusion.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.

Output in the following format:
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Answer: YES|NO|<short single phrase>

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Answer: Stein Erik Ulvund

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Answer: 13 September 1931

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Answer: NO

Question: {THE_QUESTION}
""",

    "triple_is_true_logprob": """Provide the answer and the supporting triples that lead to your conclusion.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
For each triple and for the final answer, also output whether you are confident the item is correct as True or False.

Output in the following format:
Triple 1: (Subject, Relation, Object) True/False
Triple 2: (Subject, Relation, Object) True/False
...
Answer: YES|NO|<short single phrase> True/False

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952) [confidence]
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969) [confidence]
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia) [confidence]
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931) [confidence]
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Triple 1: (The Silver Trail, director, Bernard B. Ray) [confidence]
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti) [confidence]
Triple 3: (Bernard B. Ray, country of citizenship, American) [confidence]
Triple 4: (Márton Keleti, country of citizenship, Hungarian) [confidence]
Answer: NO [confidence]

Question: {THE_QUESTION}
""",

    # C4
    "triple_verb_1s_top_1": """Provide the answer and the supporting triples that lead to your conclusion.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
Also provide your confidence (0.00–1.00, two decimals) for each triple and for the final answer.

Output in the following format:
Triple 1: (Subject, Relation, Object) 0.00–1.00
Triple 2: (Subject, Relation, Object) 0.00–1.00
...
Answer: YES|NO|<short single phrase> 0.00–1.00

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952) [confidence]
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969) [confidence]
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia) [confidence]
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931) [confidence]
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Triple 1: (The Silver Trail, director, Bernard B. Ray) [confidence]
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti) [confidence]
Triple 3: (Bernard B. Ray, country of citizenship, American) [confidence]
Triple 4: (Márton Keleti, country of citizenship, Hungarian) [confidence]
Answer: NO [confidence]

Question: {THE_QUESTION}
""",

    # C3 A+E+P_ans
    "triple_verb_1s_top_1_ansconf": """Provide the answer and the supporting triples that lead to your conclusion.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
Also provide your confidence (0.00–1.00, two decimals) for the final answer.

Output in the following format:
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Answer: YES|NO|<short single phrase> 0.00–1.00

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Answer: NO [confidence]

Question: {THE_QUESTION}
""",

    # C2  A+E only
    "triple_verb_1s_top_1_noconf": """Provide the answer and the supporting triples that lead to your conclusion.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.

Output in the following format:
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Answer: YES|NO|<short single phrase>

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Answer: Stein Erik Ulvund

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Answer: 13 September 1931

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Answer: NO

Question: {THE_QUESTION}
""",
    # C4 A+E+P_ans+P_evid (with CoT)
    "triple_cot_level_baseline": """Provide the answer and the supporting triples that lead to your conclusion.
First, show your reasoning process step by step, then output the supporting triples and the final answer.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
Finally, report your confidence (0.00–1.00, two decimals) in your overall reasoning process (Thought + Triples) and, separately, your confidence in the answer.

Output in the following format:
Thought: [reasoning process]
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Overall reasoning confidence: 0.00–1.00
Answer: YES|NO|<short single phrase> 0.00–1.00

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Thought: Compare both dates of birth and choose the earlier one.
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Overall reasoning confidence: [confidence]
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Thought: Identify his father and retrieve his date of death.
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Overall reasoning confidence: [confidence]
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Thought: Determine each film’s director and compare their countries of citizenship.
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Overall reasoning confidence: [confidence]
Answer: NO [confidence]

Question: {THE_QUESTION}
""",
    # C4 A+E+P_ans+P_evid (with CoT)
    "triple_verb_1s_cot": """Provide the answer and the supporting triples that lead to your conclusion.
First, briefly show your reasoning process, then output the supporting triples and the answer.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
Also provide your confidence (0.00–1.00, two decimals) for each triple and for the final answer.

Output in the following format:
Thought: [reasoning process]
Triple 1: (Subject, Relation, Object) 0.00–1.00
Triple 2: (Subject, Relation, Object) 0.00–1.00
...
Answer: YES|NO|<short single phrase> 0.00–1.00

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Thought: Compare both dates of birth and choose the earlier one.
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952) [confidence]
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969) [confidence]
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Thought: Identify his father and retrieve his date of death.
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia) [confidence]
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931) [confidence]
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Thought: Determine each film’s director and compare their countries of citizenship.
Triple 1: (The Silver Trail, director, Bernard B. Ray) [confidence]
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti) [confidence]
Triple 3: (Bernard B. Ray, country of citizenship, American) [confidence]
Triple 4: (Márton Keleti, country of citizenship, Hungarian) [confidence]
Answer: NO [confidence]

Question: {THE_QUESTION}
""",

    # C3 A+E+P_ans (with CoT, answer confidence only)
    "triple_verb_1s_cot_ansconf": """Provide the answer and the supporting triples that lead to your conclusion.
First, briefly show your reasoning process, then output the supporting triples and the answer.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
Also provide your confidence (0.00–1.00, two decimals) for the final answer.

Output in the following format:
Thought: [reasoning process]
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Answer: YES|NO|<short single phrase> 0.00–1.00

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Thought: Compare both dates of birth and choose the earlier one.
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Answer: Stein Erik Ulvund [confidence]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Thought: Identify his father and retrieve his date of death.
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Answer: 13 September 1931 [confidence]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Thought: Determine each film’s director and compare their countries of citizenship.
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Answer: NO [confidence]

Question: {THE_QUESTION}
""",

    # C2 CoT, no confidences
    "triple_verb_1s_cot_noconf": """Provide the answer and the supporting triples that lead to your conclusion.
First, briefly show your reasoning process, then output the supporting triples and the answer.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.

Output in the following format:
Thought: [reasoning process]
Triple 1: (Subject, Relation, Object)
Triple 2: (Subject, Relation, Object)
...
Answer: YES|NO|<short single phrase>

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Thought: Compare both dates of birth and choose the earlier one.
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952)
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969)
Answer: Stein Erik Ulvund

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Thought: Identify his father and retrieve his date of death.
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia)
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931)
Answer: 13 September 1931

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Thought: Determine each film’s director and compare their countries of citizenship.
Triple 1: (The Silver Trail, director, Bernard B. Ray)
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti)
Triple 3: (Bernard B. Ray, country of citizenship, American)
Triple 4: (Márton Keleti, country of citizenship, Hungarian)
Answer: NO

Question: {THE_QUESTION}
""",

    "triple_ling_1s_human": """Provide the answer and the supporting triples that lead to your conclusion.
First, show your reasoning process, then output the supporting triples and the answer.
Triples must be in the form (Subject, Relation, Object). The subject must be an entity, and the object must be either an entity or a specific value (date, number, etc.). Use short single phrases for all fields.
For each triple and for the final answer, express your confidence using one of the following expressions: {EXPRESSION_LIST}

Output in the following format:
Thought: [reasoning process]
Triple 1: (Subject, Relation, Object) <confidence expression>
Triple 2: (Subject, Relation, Object) <confidence expression>
...
Answer: YES|NO|<short single phrase> <confidence expression>

Examples:
Question: Who was born first, Stein Erik Ulvund or Sonya Kilkenny?
Thought: Compare both dates of birth and choose the earlier one.
Triple 1: (Stein Erik Ulvund, date of birth, 11 August 1952) [confidence expression]
Triple 2: (Sonya Kilkenny, date of birth, 15 May 1969) [confidence expression]
Answer: Stein Erik Ulvund [confidence expression]

Question: When did Prince Friedrich Sigismund of Prussia (1891–1927)'s father die?
Thought: Identify his father and retrieve his date of death.
Triple 1: (Prince Friedrich Sigismund of Prussia (1891–1927), father, Prince Friedrich Leopold of Prussia) [confidence expression]
Triple 2: (Prince Friedrich Leopold of Prussia, date of death, 13 September 1931) [confidence expression]
Answer: 13 September 1931 [confidence expression]

Question: Are the directors of the films The Silver Trail and Dreams of Love – Liszt from the same country?
Thought: Determine each film’s director and compare their countries of citizenship.
Triple 1: (The Silver Trail, director, Bernard B. Ray) [confidence expression]
Triple 2: (Dreams of Love – Liszt, director, Márton Keleti) [confidence expression]
Triple 3: (Bernard B. Ray, country of citizenship, American) [confidence expression]
Triple 4: (Márton Keleti, country of citizenship, Hungarian) [confidence expression]
Answer: NO [confidence expression]

Question: {THE_QUESTION}
""",
}

# Japanese prompt templates

TRIPLE_PROMPT_TEMPLATES_JP = {
    "triple_label_prob": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。

以下の形式で出力してください：
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
回答: YES|NO|<短い単一フレーズ>

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本)
回答: Wii U

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
回答: 18世紀

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
回答: YES

質問: {THE_QUESTION}
""",
    "triple_is_true_prob": """あなたが提供した各トリプルについて、その正確さに対する確信度を評価してください。

以下の形式で回答してください：
回答確信度: <(A) 正しい または(B) 間違っている、追加のコメントは一切なく；(A) または(B)だけを！>
トリプル1確信度: <(A) 正しい または(B) 間違っている、追加のコメントは一切なく；(A) または(B)だけを！>
トリプル2確信度: <(A) 正しい または(B) 間違っている、追加のコメントは一切なく；(A) または(B)だけを！>
...""",
    "triple_is_true_prob_a": """あなたが提供した回答について、その正確さに対する確信度を評価してください。

以下の形式で回答してください：
回答確信度: <(A) 正しい または(B) 間違っている、追加のコメントは一切なく；(A) または(B)だけを！>
...""",
    "triple_is_true_logprob_a": """あなたが提供した回答について、その正確さをTrue/Falseで評価してください。

以下の形式で回答してください：
回答確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
...""",
    "triple_is_true_logprob": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
各トリプルと回答の正確さに対する確信度をTrueまたはFalseで評価してください。

以下の形式で出力してください：
トリプル1: (主語, 関係, 目的語) True/False
トリプル2: (主語, 関係, 目的語) True/False
...
回答: YES|NO|<短い単一フレーズ> True/False

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル) [確信度]
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本) [確信度]
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派) [確信度]
トリプル2: (ワッハーブ派，生まれた世紀，18世紀) [確信度]
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
トリプル1: (反町隆史，職業，元ファッションモデル) [確信度]
トリプル2: (成田凌，職業，ファッションモデル) [確信度]
回答: YES [確信度]

質問: {THE_QUESTION}
""",
    # 1ステップでトップ1の回答とトリプル、確率を言語化
#     "triple_verb_1s_top_1": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。各トリプルについて、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

# 質問: {THE_QUESTION}

# 以下の形式で回答してください：
# トリプル1: (エンティティ1，関係，エンティティ2)
# トリプル2: (エンティティ1，関係，エンティティ2)
# ...
# 回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>

# トリプル1確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# トリプル2確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# ...
# 回答確信度: <回答全体の正確さの確率（0.0〜1.0）>
# """,
# sepconf
#     "triple_verb_1s_top_1": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。各トリプルについて、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

# 質問: {THE_QUESTION}

# 以下の形式で<追加のコメントは一切なく、トリプル、回答のみ、確信度は確率の数値のみを>出力してください。：
# トリプル1: (エンティティ1，関係，エンティティ2) 0.0〜1.0の間の確率
# トリプル2: (エンティティ1，関係，エンティティ2) 0.0〜1.0の間の確率
# ...
# 回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！> 0.0〜1.0の間の確率
# """,
# fix4eval
#     "triple_verb_1s_top_1": """次の質問に答え、その根拠トリプルを出力せよ  
# 各トリプルと回答の正確さに対する確信度を 0.00–1.00（小数 2 桁）で示すこと。
# ★トリプルは (主語, 関係, 目的語) の形式。主語・目的語は短い単一フレーズのエンティティまたは具体的な値（目的語）のみ。説明・注釈・Bool値（はい/いいえ 等）は禁止。比較専用の関係（例：先／大きい／高い など）は禁止で、必ず日付・数値・カテゴリなど具体的に示すこと。★
# **重要**: 補足的・背景的・質問内から読み取れる自明な情報は含めず、回答に直接必要な情報のみ必要最低限のトリプルを出力すること。

# 出力フォーマット（厳守）：
# トリプル1: (主語, 関係, 目的語) 0.00–1.00
# トリプル2: (主語, 関係, 目的語) 0.00–1.00
# ...
# 回答: YES|NO|<短い単一フレーズ> 0.00–1.00

# 質問: {THE_QUESTION}
# """,
# fix4evalfs(fewshot)
# C4
    "triple_verb_1s_top_1": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
各トリプルと回答の正確さに対する確信度（0.00–1.00,小数2桁）も示すこと。

以下の形式で出力してください：
トリプル1: (主語, 関係, 目的語) 0.00–1.00
トリプル2: (主語, 関係, 目的語) 0.00–1.00
...
回答: YES|NO|<短い単一フレーズ> 0.00–1.00

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル) [確信度]
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本) [確信度]
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派) [確信度]
トリプル2: (ワッハーブ派，生まれた世紀，18世紀) [確信度]
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
トリプル1: (反町隆史，職業，元ファッションモデル) [確信度]
トリプル2: (成田凌，職業，ファッションモデル) [確信度]
回答: YES [確信度]

質問: {THE_QUESTION}
""",
# C3 A+E+P_ans
    "triple_verb_1s_top_1_ansconf": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
回答の正確さに対する確信度（0.00–1.00,小数2桁）も示すこと。

以下の形式で出力してください：
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
回答: YES|NO|<短い単一フレーズ> 0.00–1.00

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本)
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
回答: YES [確信度]

質問: {THE_QUESTION}
""",
# C2  A+Eのみ
   "triple_verb_1s_top_1_noconf": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。

以下の形式で出力してください：
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
回答: YES|NO|<短い単一フレーズ>

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本)
回答: Wii U

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
回答: 18世紀

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
回答: YES

質問: {THE_QUESTION}
""",
    "triple_verb_1s_top_1_a": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。回答の正確さに対するあなたの確信度（0.0〜1.0）も示してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...
回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <回答全体の正確さの確率（0.0〜1.0）>
""",
    # 1ステップで複数の回答候補とトリプルセット、確率を言語化
    "triple_verb_1s_top_2": """次の質問に対する回答と、その結論に至る可能性のある2つの代替トリプルセットを提供してください。各セットについて、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

質問: {THE_QUESTION}

以下の形式で回答してください：

セット1:
回答1: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答1確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル1.1: (エンティティ1，関係，エンティティ2)
トリプル1.1確信度: <トリプル1.1の正確さの確率（0.0〜1.0）>
トリプル1.2: (エンティティ1，関係，エンティティ2)
トリプル1.2確信度: <トリプル1.2の正確さの確率（0.0〜1.0）>
...

セット2:
回答2: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答2確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル2.1: (エンティティ1，関係，エンティティ2)
トリプル2.1確信度: <トリプル2.1の正確さの確率（0.0〜1.0）>
トリプル2.2: (エンティティ1，関係，エンティティ2)
トリプル2.2確信度: <トリプル2.1の正確さの確率（0.0〜1.0）>
...
""",

    # 1ステップでより多くの回答候補とトリプルセット、確率を言語化
    "triple_verb_1s_top_4": """次の質問に対する回答と、その結論に至る可能性のある4つの代替トリプルセットを提供してください。各セットについて、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>

セット1:
回答1: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答1確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル1.1: (エンティティ1，関係，エンティティ2)
トリプル1.1確信度: <トリプル1.1の正確さの確率（0.0〜1.0）>
トリプル1.2: (エンティティ1，関係，エンティティ2)
トリプル1.2確信度: <トリプル1.2の正確さの確率（0.0〜1.0）>
...

セット2:
回答2: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答2確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル2.1: (エンティティ1，関係，エンティティ2)
トリプル2.1確信度: <トリプル2.1の正確さの確率（0.0〜1.0）>
トリプル2.2: (エンティティ1，関係，エンティティ2)
トリプル2.2確信度: <トリプル2.1の正確さの確率（0.0〜1.0）>
...

セット3:
回答3: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答3確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル3.1: (エンティティ1，関係，エンティティ2)
トリプル3.1確信度: <トリプル3.1の正確さの確率（0.0〜1.0）>
トリプル3.2: (エンティティ1，関係，エンティティ2)
トリプル3.2確信度: <トリプル3.2の正確さの確率（0.0〜1.0）>
...

セット4:
回答4: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答4確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
トリプル4.1: (エンティティ1，関係，エンティティ2)
トリプル4.1確信度: <トリプル4.1の正確さの確率（0.0〜1.0）>
トリプル4.2: (エンティティ1，関係，エンティティ2)
トリプル4.2確信度: <トリプル4.1の正確さの確率（0.0〜1.0）>
...
""",

    # 2ステップ方式: 1段階目でトリプルを生成
    "triple_verb_2s_top_1": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...
""",
    "triple_verb_2s_top_k": """以下の質問に対して、{k}セットの可能性の高いトリプル集合を提供してください。各セットは質問の回答を導出するために関連するトリプルを含みます。

質問: {THE_QUESTION}

以下の形式で回答してください：

セット1:
回答1: <最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>
トリプル1.1: (エンティティ1，関係，エンティティ2)
トリプル1.2: (エンティティ1，関係，エンティティ2)
...

セット2:
回答2: <最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>
トリプル2.1: (エンティティ1，関係，エンティティ2)
トリプル2.2: (エンティティ1，関係，エンティティ2)
...

...（{k}セットまで続く）
""",

#     # 2ステップ方式: 2段階目で確率を評価
#     "triple_verb_2s_top_1_prob": """あなたが提供した各トリプルについて、その正確さに対するあなたの確信度（0.0〜1.0）を示してください。

# 以下の形式で回答してください：
# 確率1: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# 確率2: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# ...
# """,
    # Chain-of-Thoughtを使ったトリプル評価
#     "triple_verb_1s_cot": """次の質問に対する答えを考えるプロセスを段階的に示してください。確認する必要のある事実を特定し、最終的な回答とその根拠となるトリプルを提供してください。各トリプルについて、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

# 質問: {THE_QUESTION}

# 以下の形式で回答してください：
# 思考: <あなたの思考過程を一文で段階的に説明>
# トリプル1: (エンティティ1，関係，エンティティ2)
# トリプル2: (エンティティ1，関係，エンティティ2)
# ...

# 回答: <あなたの最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>

# 回答確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
# トリプル1確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# トリプル2確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
# ...
# """,
# C4 A+E+P_ans+P_evid
    "triple_cot_level_baseline": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
まず思考過程を段階的に示し、その後で根拠トリプルと回答を出力してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
最後に、あなたの推論過程全体（思考とトリプル）に対する確信度（0.00–1.00,小数2桁）と、回答の確信度を別々に示すこと。

以下の形式で出力してください：
思考: [推論過程]
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
推論全体の確信度: 0.00–1.00
回答: YES|NO|<短い単一フレーズ> 0.00–1.00

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
思考: 両ゲーム機の日本での同時発売ソフト数を比較する必要がある。
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本)
推論全体の確信度: [確信度]
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
思考: ビン・ラーディンの宗派を特定し、その成立時期を調べる必要がある。
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
推論全体の確信度: [確信度]
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
思考: 両者の職業経歴を確認する必要がある。
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
推論全体の確信度: [確信度]
回答: YES [確信度]

質問: {THE_QUESTION}
""",
    "triple_verb_1s_cot": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
まず思考過程を端的に示し、その後で根拠トリプルと回答を出力してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
各トリプルと回答の正確さに対する確信度（0.00–1.00,小数2桁）も示すこと。

以下の形式で出力してください：
思考: [推論過程]
トリプル1: (主語, 関係, 目的語) 0.00–1.00
トリプル2: (主語, 関係, 目的語) 0.00–1.00
...
回答: YES|NO|<短い単一フレーズ> 0.00–1.00

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
思考: 両ゲーム機の日本での同時発売ソフト数を比較する必要がある。
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル) [確信度]
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本) [確信度]
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
思考: ビン・ラーディンの宗派を特定し、その成立時期を調べる必要がある。
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派) [確信度]
トリプル2: (ワッハーブ派，生まれた世紀，18世紀) [確信度]
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
思考: 両者の職業経歴を確認する必要がある。
トリプル1: (反町隆史，職業，元ファッションモデル) [確信度]
トリプル2: (成田凌，職業，ファッションモデル) [確信度]
回答: YES [確信度]

質問: {THE_QUESTION}
""",
# C3 A+E+P_ans
    "triple_verb_1s_cot_ansconf": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
まず思考過程を端的に示し、その後で根拠トリプルと回答を出力してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
回答の正確さに対する確信度（0.00–1.00,小数2桁）も示すこと。

以下の形式で出力してください：
思考: [推論過程]
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
回答: YES|NO|<短い単一フレーズ> 0.00–1.00

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
思考: 両ゲーム機の日本での同時発売ソフト数を比較する必要がある。
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本) 
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
思考: ビン・ラーディンの宗派を特定し、その成立時期を調べる必要がある。
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
思考: 両者の職業経歴を確認する必要がある。
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
回答: YES [確信度]

質問: {THE_QUESTION}
""",
# C2 C2 noconf
    "triple_verb_1s_cot_noconf": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
まず思考過程を端的に示し、その後で根拠トリプルと回答を出力してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。

以下の形式で出力してください：
思考: [推論過程]
トリプル1: (主語, 関係, 目的語)
トリプル2: (主語, 関係, 目的語)
...
回答: YES|NO|<短い単一フレーズ>

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
思考: 両ゲーム機の日本での同時発売ソフト数を比較する必要がある。
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル)
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本)
回答: Wii U

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
思考: ビン・ラーディンの宗派を特定し、その成立時期を調べる必要がある。
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派)
トリプル2: (ワッハーブ派，生まれた世紀，18世紀)
回答: 18世紀

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
思考: 両者の職業経歴を確認する必要がある。
トリプル1: (反町隆史，職業，元ファッションモデル)
トリプル2: (成田凌，職業，ファッションモデル)
回答: YES

質問: {THE_QUESTION}
""",

# sepconf
#     "triple_verb_1s_cot": """次の質問に対をする答えを考えるプロセスを段階的に示してください。確認する必要のある事実を特定し、その根拠となるトリプルと回答を提供してください。各トリプルと回答について、その正確さに対するあなたの確信度（0.0〜1.0）も示してください。

# 質問: {THE_QUESTION}

# 以下の形式で<追加のコメントは一切なく、トリプル、回答のみ、確信度は確率の数値のみを>出力してください。：
# 思考: <あなたの思考過程を一文で段階的に説明>
# トリプル1: (エンティティ1，関係，エンティティ2) 0.0〜1.0の間の確率
# トリプル2: (エンティティ1，関係，エンティティ2) 0.0〜1.0の間の確率
# ...
# 回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！> 0.0〜1.0の間の確率
# """,
    "triple_verb_1s_cot_a": """次の質問に対する答えを考えるプロセスを段階的に示してください。確認する必要のある事実を特定し、最終的な回答とその根拠となるトリプルを提供してください。回答の正確さに対するあなたの確信度（0.0〜1.0）も示してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
思考: <あなたの思考過程を一文で段階的に説明>
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...

回答: <あなたの最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>

回答確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
...
""",
    # Chain-of-Thoughtを使ったトリプル評価
    "triple_verb_1s_cot_is_true": """次の質問に対する答えを考えるプロセスを段階的に示してください。確認する必要のある事実を特定し、最終的な回答とその根拠となるトリプルを提供してください。その後、回答と各トリプルについて、その正確さをTrue/Falseで評価してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
思考: <あなたの思考過程を一文で段階的に説明>
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...

回答: <あなたの最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>

トリプル1確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
トリプル2確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
回答確信度: <True または False、追加のコメントは一切なく；True またはFalseだけを！>
...
""",
    "triple_verb_2s_cot": """以下の質問について、Chain-of-Thoughtプロセスを使って回答してください。まず思考過程を示し、次に回答とその根拠となる複数のトリプルを示してください。

質問: {THE_QUESTION}

以下の形式で回答してください：
思考過程: <あなたの思考過程を一文で段階的に説明>
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...
回答: <あなたの最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>

""",
    "triple_verb_2s_prob": """あなたが提供した各トリプルについて、その正確さに対する確信度を評価してください。

以下の形式で回答してください：
トリプル1確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
トリプル2確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
...
回答確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>

""",
    "triple_verb_2s_prob_a": """あなたが提供した回答について、その正確さに対する確信度を評価してください。

以下の形式で回答してください：
回答確信度: <あなたの回答が正しい確率（0.0から1.0の間）、追加のコメントは一切なく；確率だけを！>
...
""",
    # 1ステップでトップ1の回答とトリプル、確率を言語化
    "triple_verb_3s": """次の質問に答えるために必要な根拠となるトリプルを2つ以上必要な個数挙げてください。

質問: {THE_QUESTION}

以下の形式で回答してください：
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
...
""",
    "triple_verb_3s_triple_prob": """あなたが提供した各トリプルについて、その正確さに対する確信度を評価してください。

以下の形式で回答してください：
トリプル1確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
トリプル2確信度: <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
...
""",
    #TRIPLE_AND_CONFIDENCES:
    #トリプル1: [主語1、述語1、目的語1] - 確信度: {確信度1}
    #トリプル2: [主語2、述語2、目的語2] - 確信度: {確信度2}
    #トリプル3: [主語3、述語3、目的語3] - 確信度: {確信度3}
    "triple_verb_3s_answer_prob": """以下のトリプルとその確信度に基づいて、質問に対する最終的な答えとその確信度を出力してください。
質問: {THE_QUESTION}
{TRIPLE_AND_CONFIDENCES}

最終回答:<最終的な回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <回答全体の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
計算方法: <最終確信度の計算方法を一文で説明>
...
""",
    # 1ステップでトップ1の回答とトリプル、確率を言語化 triple_verb_1s_top_1の後に実行
    #ANSWER_TRIPLE_AND_CONFIDENCES:
    # トリプル1: [主語1、述語1、目的語1] - 確信度: 0.85
    # トリプル2: [主語2、述語2、目的語2] - 確信度: 0.73
    # トリプル3: [主語3、述語3、目的語3] - 確信度: 0.91
    # 最終回答: {回答} - 確信度: 0.82
    "triple_verb_meta": """あなたが先ほど生成した根拠トリプルとその確信度について、再評価してください。

質問: {THE_QUESTION}

【最初の評価】
{ANSWER_TRIPLE_AND_CONFIDENCES}

【再評価】
以上の評価が適切かどうか検討し、確信度を調整する必要があれば理由とともに説明してください。

トリプル1の確信度:調整後 = <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
トリプル2の確信度:調整後 = <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
...
最終確信度:調整後 = <0.0〜1.0の間の確率、追加のコメントは一切なく；確率だけを！>
調整理由: <調整する必要があれば理由を一文で説明>
""",
    "triple_verb_2s_top_k_prob": """あなたが提供した各セットとその中の個々のトリプルについて、正確さに対する確信度を評価してください。

以下の形式で回答してください：

回答1確信度: <回答全体の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
トリプル1.1確信度: <トリプル1.1の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
トリプル1.2確信度: <トリプル1.2の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
...

回答2確信度: <回答全体の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
トリプル2.1確信度: <トリプル2.1の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
トリプル2.2確信度: <トリプル2.2の正確さの確率（0.0〜1.0）、追加のコメントは一切なく；確率だけを！>
...
""",

    # 言語表現を使用してトリプルの確信度を表現
#     "triple_ling_1s_human": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。各トリプルについて、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

# 質問: {THE_QUESTION}

# 以下の形式で回答してください：
# トリプル1: (エンティティ1，関係，エンティティ2)
# 確信度1: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
# トリプル2: (エンティティ1，関係，エンティティ2)
# 確信度2: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
# 回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
# 回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
# ...
# """,
#     "triple_ling_1s_human": """次の質問に対して、その根拠となるトリプルと回答を提供してください。回答はYesかNoまたはフレーズで出力してください。また、各トリプルと回答について、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

# 質問: {THE_QUESTION}

# 以下の形式で<追加のコメントは一切なく、トリプル、回答、確信度のみを>出力してください。：
# トリプル1: (エンティティ1，関係，エンティティ2) 上記の確信度のフレーズ
# トリプル2: (エンティティ1，関係，エンティティ2) 上記の確信度のフレーズ
# ...
# 回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！> 上記の確信度のフレーズ
# """,
    "triple_ling_1s_human": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。
まず思考過程を示し、その後で根拠トリプルと回答を出力してください。
トリプルは (主語, 関係, 目的語) の形式。主語はエンティティ、目的語はエンティティまたは具体的な値（日付、数値等）とし、いずれも短い単一フレーズとする。
各トリプルと回答の正確さに対する確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

以下の形式で出力してください：
思考: [推論過程]
トリプル1: (主語, 関係, 目的語) ほぼ確実|非常に可能性が高い| ... |ほぼ可能性がない
トリプル2: (主語, 関係, 目的語) ほぼ確実|非常に可能性が高い| ... |ほぼ可能性がない
...
回答: YES|NO|<短い単一フレーズ> ほぼ確実|非常に可能性が高い| ... |ほぼ可能性がない

例：
質問: Wii Uとニンテンドーゲームキューブ、日本での本体と同時発売のゲームソフト数が多いのはどちらでしょう？
思考: 両ゲーム機の日本での同時発売ソフト数を比較する必要がある。
トリプル1: (Wii U，日本での本体と同時発売のソフト数，11タイトル) [確信度]
トリプル2: (ニンテンドーゲームキューブ，日本での本体と同時発売のソフト数，2本) [確信度]
回答: Wii U [確信度]

質問: ウサーマ・ビン・ラーディンの信仰するイスラム教の宗派は何世紀に起こりましたか？
思考: ビン・ラーディンの宗派を特定し、その成立時期を調べる必要がある。
トリプル1: (ウサーマ・ビン・ラーディン，信仰するイスラム教の宗派，ワッハーブ派) [確信度]
トリプル2: (ワッハーブ派，生まれた世紀，18世紀) [確信度]
回答: 18世紀 [確信度]

質問: 反町隆史と成田凌、どちらもファッションモデルの経験がありますか？
思考: 両者の職業経歴を確認する必要がある。
トリプル1: (反町隆史，職業，元ファッションモデル) [確信度]
トリプル2: (成田凌，職業，ファッションモデル) [確信度]
回答: YES [確信度]

質問: {THE_QUESTION}
""",
    "triple_ling_1s_human_a": """次の質問に対する回答と、その結論に至った根拠となるトリプルを提供してください。各トリプルについて、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

質問: {THE_QUESTION}

以下の形式で回答してください：
トリプル1: (エンティティ1，関係，エンティティ2)
トリプル2: (エンティティ1，関係，エンティティ2)
回答: <あなたの回答、できるだけ短く；完全な文ではなく、回答のみを！>
回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>

...
""",
    # 言語表現を使用してトリプルの確信度を表現
    "triple_ling_2s_human": """あなたが提供した回答と各トリプルについて、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

以下の形式で回答してください：
回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
トリプル1確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
トリプル2確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
...
""",

    # 言語表現を使用してトリプルの確信度を表現
    "triple_ling_2s_human_a": """あなたが提供した回答について、その正確さに対するあなたの確信度を以下の表現のいずれかを使って示してください： {EXPRESSION_LIST}

以下の形式で回答してください：
回答確信度: <確信度の表現、追加のコメントは一切なく；短いフレーズだけを！>
...
"""

}

# トリプル評価のプロンプト
TRIPLE_EVALUATION_PROMPT = """Evaluate the accuracy of the following predicted triple for the given question:

Question: {THE_QUESTION}

Predicted Triple: {PRED_TRIPLE}
Gold Triples: {GOLD_TRIPLES}

Rate the predicted triple's accuracy on a scale:
1.0 - Completely correct (subject, relation, and object all match a gold triple)
0.5 - Partially correct (subject and relation match but object has minor errors, OR subject and object match but relation has minor errors)
0.0 - Incorrect (major errors)

Provide your rating as a single number (1.0, 0.5, or 0.0):
"""

# トリプル評価のプロンプト（日本語版）
# PRED_TRIPLES:
# 予測トリプル1:  (エンティティ1，関係，エンティティ2)
# 予測トリプル2:  (エンティティ1，関係，エンティティ2)
TRIPLE_EVALUATION_PROMPT_JP_bak = """次の質問に対する予測トリプルの正確さを以下の基準で評価してください：
1.0 - 完全に正確（（主体, 関係, 対象）がすべて正解トリプルと一致する、または意味的に合致する、対象の粒度が異なるが質問に回答するには十分な情報を含む）
0.5 - 部分的に正確（主体と対象は一致するが関係に軽微な誤りがある※対象が異なる実体を指す場合は不正確）
0.0 - 不正確（対象が意味的に一致しない）

評価ポイント:
・ 1つのトリプルの情報が2つのトリプルに分割されている場合は許容する
・ 表記が異なっても質問に適切なら粒度の違いを許容するが、問われているより上位概念での回答は不正解
・ 日付は年月日が根拠として必要な質問では完全一致が必要であるが、質問への回答に十分であれば年や年月のみでも許容
・ 数値は四捨五入を許容

質問: ${THE_QUESTION}
正解トリプル: ${GOLD_TRIPLES}
予測トリプル:${PRED_TRIPLES}

評価を単一の数字（1.0、0.5、または0.0）で、以下の形式で回答してください：
予測トリプル1評価: <評価の単一の数字、追加のコメントは一切なく；数字のみを！>
予測トリプル2評価: <評価の単一の数字、追加のコメントは一切なく；数字のみを！>
...
"""

TRIPLE_EVALUATION_PROMPT_JP_bak2 = """次の質問に対する予測トリプルの正確さを以下の基準で評価してください：
1.0 - 完全に正確（（主体, 関係, 対象）がすべて正解トリプルと一致する、または意味的に合致する、対象の粒度が異なるが質問に回答するには十分な情報を含む）
0.5 - 部分的に正確（主体と対象は一致するが関係に軽微な誤りがある※対象が異なる実体を指す場合は不正確）、または情報は正しいが不要である
0.0 - 不正確（対象が意味的に一致しない）

評価ポイント:
・ 1つのトリプルの情報が2つのトリプルに分割されている場合は許容する
・ 正解トリプルに含まれていないトリプルは、あってもよい情報であれば1.0、不要だが正確であれば0.5、それ以外は0.0
・ 表記が異なっても質問に適切なら粒度の違いを許容するが、問われているより上位概念での回答は不正解
・ 日付は年月日が根拠として必要な質問では完全一致が必要であるが、質問への回答に十分であれば年や年月のみでも許容
・ 数値は四捨五入を許容

質問: ${THE_QUESTION}
正解トリプル: ${GOLD_TRIPLES}
予測トリプル:${PRED_TRIPLES}

評価を単一の数字（1.0、0.5、または0.0）で、以下の形式で回答してください：
予測トリプル1評価: <評価の単一の数字、追加のコメントは一切なく；数字のみを！>
予測トリプル2評価: <評価の単一の数字、追加のコメントは一切なく；数字のみを！>
...
"""

# bak20250709
# TRIPLE_EVALUATION_PROMPT_JP = """以下の手順で、予測トリプルと正解トリプルを一括評価し、**純粋なJSON配列のみ**を出力してください。

# 1. 予測⇔正解の最適マッチング  
#    - ハンガリアン法で1対1対応を決定  
#    - チェイン合成（分割トリプルの結合）を同値とみなす  

# 2. スコア付与  
#    - 1.0: 主語・述語・目的語が完全一致、または以下の意味的合致ケース
#      • 同義語・言い換え 
#      • 数値／日時の丸めや粒度違い
#      • チェイン合成による同値（分割トリプルの組み合わせ）
#    - 0.5: 主語・目的語一致、述語に軽微誤り
#      • 目的語に必要粒度を超える誤った詳細を含む（例：正解が「2000年」のところ予測が「2000年1月1日」で1月1日が誤り）
#    - 0.0: 意味的に不一致  

# 3. 追加例外  
#    - ゴールドにない予測トリプルは「許容情報→1.0／不要だが正確→0.5／その他→0.0」  
#    - 表記ゆれ・同義語OK、必要な情報より上位概念はNG  
#    - 日付は必要粒度で完全一致、十分な部分一致を許容  
#    - 数値は四捨五入許容  

# 4. 出力スキーマ  
# ```json
# [
#   {
#     "type": "pred",     // "pred" or "gold"
#     "index": 1,         // 必ずもとの予測トリプルの記載順で1から要素番号をふる
#     "triple":  "（a, b, c）",  //（主体, 関係, 対象）
#     "matched_index": 3, // 対応する相手のindex、未対応はnull
#     "score": 1.0        // 0.0, 0.5, または 1.0
#   },
#   {
#     "type": "pred",
#     "index": 2,
#     "triple":  "（d, e, f）",
#     "matched_index": null,
#     "score": 0.0
#   },
#   {
#     "type": "gold",
#     "index": 3,        // 必ずもとの正解トリプルの記載順でpredの最後のindex+1から要素番号をふる
#     "triple": "（a, b, c+）",
#     "matched_index": 1,
#     "score": 1.0
#   },
#   …
# ]```

# 質問: ${THE_QUESTION}
# 正解トリプル: ${GOLD_TRIPLES}
# 予測トリプル: ${PRED_TRIPLES}
# """

# fix4eval
TRIPLE_EVALUATION_PROMPT_JP = """**返答は JSON 配列のみ。説明・コメント・前後の文は禁止。**

【タスク】
予測トリプル (pred) と正解トリプル (gold) をハンガリアン法で1対1マッチングし、各ペアにスコア（1.0 or 0.0）を付与してJSON配列で返す。

【スコア基準】
**1.0:** 以下のいずれかを満たす
- 完全一致：主語・関係・目的語が全て一致
- 意味的合致：同義語・言い換えで本質的に同じ（例：俳優↔役者）
- 情報等価：表現は異なるが、質問回答に必要な情報が完全に同じ。曖昧なら0.0
  • 例：国のみが必要な質問の場合に、(A,出版された国,日本) ≈ (A,ジャンル,日本の漫画)
  • 例：年のみが必要な質問の場合に、(A,出版日,2001年12月10日) ≈ (A,連載開始年,2001年)
- チェイン合成：2hop以内の推論で導出可能
  • 例：(A,r1,B)+(B,r2,C) ⇔ (A,r*,C)
- 正確な追加情報：goldにない予測でも正確で関連する情報

**0.0:** 上記を満たさない、または以下の形式違反
- 主語・目的語がエンティティでない（Bool値、文章等）
- トリプル形式 (主語エンティティ, 関係, 目的語エンティティ) に従わない

迷った場合は 0.0 を選択。

【出力スキーマ】出力順序： pred の1,2,... → gold の1,2,... の順
[
  {
    "type": "pred",          // "pred" or "gold"
    "index": 1,              // 元のリスト順に 1,2,…
    "triple": "(a,b,c)",     // (主語エンティティ, 関係, 目的語エンティティ)
    "matched_index": 3,      // 相手 index。対応無しは null
    "score": 1.0             // 1.0 または 0.0
  },
  …
]

【入力】
質問: ${THE_QUESTION}
正解トリプル: ${GOLD_TRIPLES}
予測トリプル: ${PRED_TRIPLES}
"""
# TRIPLE_EVALUATION_PROMPT_JP = """
# 質問: ${THE_QUESTION}

# 正解トリプル: ${GOLD_ANSWER}
# 予測トリプル: ${PRED_ANSWER}

# 上記の予測トリプルは正解トリプルと意味的に同等であるかどうかのスコアを
# 評価してください。

# 評価ポイント:
# 1. 表記が異なっても、指している実体が同じであれば正解とします（例: 「アップル」と「Apple」など）
# 2. 順序がトリプル形式（主体, 関係, 対象）に従っていれば正解とします
# 3. トリプルの表現が異なる場合も、トリプル全体で示す情報が質問に回答するために十分であれば正解とします

# 予測トリプルは正解と意味的に同等ですか？ スコアを0（不正解）、0.5（部分的に正解）、1（完全に正解）で評価し、理由を説明してください。
# """