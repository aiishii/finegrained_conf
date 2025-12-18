import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# Predefined mappings
TRUE_FALSE_MAP = {
    "true": 0.8, "false": 0.2,
    "(a)": 0.8, "(b)": 0.2, "a": 0.8, "b": 0.2,
    "はい": 0.8, "いいえ": 0.2,
}
LING_EN = {
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
LING_JA = {
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
conf_pattern_en = "|".join(map(re.escape, LING_EN.keys()))
conf_pattern_ja = "|".join(map(re.escape, LING_JA.keys()))
# Merge for efficient search
LING_MAP = {**{k.lower(): v for k, v in LING_EN.items()},
            **{k.lower(): v for k, v in LING_JA.items()}}

conf_pattern = "|".join(map(re.escape, LING_MAP.keys()))

# Utility functions
def to_prob(txt: str) -> Optional[float]:
    """
    Convert text to probability using decision tree:
    1) Numeric value → float
    2) True/False variants → TRUE_FALSE_MAP
    3) Linguistic expressions → LING_MAP (exact match)
    """
    if txt is None:
        return None
    t = txt.strip().lower()
    # 1. Numeric value
    try:
        return float(t)
    except ValueError:
        pass
    # 2. True/False / A/B
    if t in TRUE_FALSE_MAP:
        return TRUE_FALSE_MAP[t]
    # 3. Linguistic expression
    if t in LING_MAP:
        return LING_MAP[t]
    return None

# Data structures
@dataclass
class TripleSet:
    triples: List[str] = field(default_factory=list)
    triple_conf: List[Optional[float]] = field(default_factory=list)
    set_conf: Optional[float] = None

@dataclass
class ParseResult:
    answers: List[str] = field(default_factory=list)
    answer_conf: List[Optional[float]] = field(default_factory=list)
    triple_sets: Dict[int, TripleSet] = field(default_factory=dict)
    thinking: Optional[str] = None

# Main parser
# Label pattern definitions
LABELS = {
    # English
    "answer":   r"^(guess|answer)\s*[:：]\s*(?P<body>.*)",
    "answer_w_conf":   r"^(?:guess|answer)\s*[:：]\s*(?P<body>.+?)\s+\[?(?P<conf>\d+(?:\.\d+)?)\]?(?:\s*)$",
    "ans_idx":  r"^(g|answer)\s*(?P<idx>\d+)\s*[:：]\s*(?P<body>.*)",
    "ans_idx_w_conf": r"^(?:g|answer)\s*(?P<idx>\d+)\s*[:：]\s*(?P<body>.+?)\s+(?P<conf>\d+(?:\.\d+)?)(?:\s*)$",
    "ans_w_vconf": rf"^(guess|answer)\s*[:：]\s*(?P<body>.+?)\s+(?P<conf>{conf_pattern_en})",
    "ans_w_tf": rf"^(guess|answer)\s*[:：]\s*(?P<body>.+?)\s+(?P<conf>True|False)(?:\s*)$",
    "ans_conf": r"^(probability|answer confidence)\s*[:：]\s*\[?(?P<body>.*)\]?",
    "ansc_idx": r"^(p|probability|answer\d* confidence)\s*(?P<idx>\d*)\s*[:：]\s*(?P<body>.*)",
    # Japanese
    "ja_ans":   r"^(最終回答|回答)\s*[:：]\s*(?P<body>.*)",
    "ja_ans_i": r"^回答\s*(?P<idx>\d+)\s*[:：]\s*(?P<body>.*)",
    "ja_ans_w_conf":   r"^(最終回答|回答)\s*[:：]\s*(?P<body>.*?)\s+\[?(?P<conf>\d+(?:\.\d+)?)\]?(?:\s*)$",
    "ja_ans_i_w_conf": r"^回答\s*(?P<idx>\d+)\s*[:：]\s*(?P<body>.*?)\s+\[?(?P<conf>\d+(?:\.\d+)?)\]?(?:\s*)$",
    "ja_ans_w_vconf": rf"^(最終回答|回答)\s*[:：]\s*(?P<body>.+?)\s+\[?(?P<conf>{conf_pattern_ja})\]?(?:\s*)$",
    "ja_ans_w_tf": rf"^(最終回答|回答)\s*[:：]\s*(?P<body>.+?)\s+\[?(?P<conf>True|False)\]?(?:\s*)$",
    "ja_acf":   r"^(確率|回答確信度|最終確信度(:調整後)?)\s*[:：=]\s*(?P<body>.*)",
    "ja_acf_i": r"^(確率|回答\d*確信度)\s*(?P<idx>\d*)\s*[:：]\s*(?P<body>.*)",
    # Triple
    "triple":   r"^(triple|トリプ*ル*)\s*(?P<tid>\d+(?:\.\d+)?)\s*[:：]\s*(?P<body>.*)",
    "triple_w_conf":   r"^(?:triple|トリプ*ル*)\s*(?P<tid>\d+(?:\.\d+)?)\s*[:：]\s*(?P<body>.+?)\s+\[?(?P<conf>\d+(?:\.\d+)?)\]?(?:\s*)$",
    "triple_w_vconf":  rf"^(?:triple|トリプ*ル*)\s*(?P<tid>\d+(?:\.\d+)?)\s*[:：]\s*(?P<body>.*?)\s+\[?(?P<conf>{conf_pattern})\]?(?:\s*)$",
    "triple_w_tf":  rf"^(?:triple|トリプ*ル*)\s*(?P<tid>\d+(?:\.\d+)?)\s*[:：]\s*(?P<body>.*?)\s+\[?(?P<conf>True|False)\]?(?:\s*)$",
    "t_conf":   r"^(?:(?:トリプ*ル*|triple)?\s*(?P<tid>\d+(?:\.\d+)?)\s*(?:confidence|確信度)|(?:confidence|確信度)\s*(?P<tid2>\d+(?:\.\d+)?))\s*[:：]\s*(?P<body>.*)",
    "t_conf_adj": r"^(?:トリプ*ル*|triple)?\s*(?P<tid>\d+(?:\.\d+)?)\s*の確信度(:調整後)?\s*=\s*(?P<body>.*)",
    # Set headers
    "set_en":   r"^set\s*(?P<sid>\d+)\s*[:：]?",
    "set_ja":   r"^セット\s*(?P<sid>\d+)\s*[:：]?",
    # Thinking
    "think":    r"^(explanation|thinking|思考|思考過程)\s*[:：]\s*(?P<body>.*)",
    "think_conf":   r"^(推論全体の確信度|Overall reasoning confidence)\s*[:：]\s*(?P<body>.*)",
}

# Compile for performance
PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in LABELS.items()}

def parse_output(raw: str) -> ParseResult:
    """
    Single entry point that works with any prompt output format
    """
    res = ParseResult()
    current_set = 1
    ts = None

    if not raw or not raw.strip():
        return res

    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # Set headers
        m = PATTERNS["set_en"].match(ln) or PATTERNS["set_ja"].match(ln)
        if m:
            current_set = int(m.group("sid"))
            continue

        # Thinking
        if not res.thinking:
            m = PATTERNS["think"].match(ln)
            if m:
                res.thinking = m.group("body").strip()
                continue


        m = PATTERNS["ans_idx_w_conf"].match(ln) or PATTERNS["ja_ans_i_w_conf"].match(ln)
        if m:
            try:
                idx = int(m.group("idx"))
                body = m.group("body").strip()
                prob = to_prob(m.group("conf"))
                _ensure_len(res.answers, idx)
                res.answers[idx-1] = body
                res.answer_conf[idx-1] = prob
            except (ValueError, IndexError) as e:
                print(f"Error parsing answer with index: {ln} - {str(e)}")
            continue

        # Answer with confidence
        m = PATTERNS["answer_w_conf"].match(ln) or PATTERNS["ja_ans_w_conf"].match(ln) or PATTERNS["ans_w_vconf"].match(ln) or PATTERNS["ja_ans_w_vconf"].match(ln) or PATTERNS["ans_w_tf"].match(ln) or PATTERNS["ja_ans_w_tf"].match(ln)
        if m:
            body = m.group("body").strip()
            prob = to_prob(m.group("conf"))
            if not res.answers:
                res.answers.append(body)
            if prob is not None:
                res.answer_conf.append(prob)

            continue

        # Answer without confidence
        m = PATTERNS["answer"].match(ln) or PATTERNS["ja_ans"].match(ln)
        if m:
            body = m.group("body").strip()
            if not res.answers:
                res.answers.append(body)
            continue

        # Answer confidence with index
        m = PATTERNS["ansc_idx"].match(ln) or PATTERNS["ja_acf_i"].match(ln)
        if m:
            try:
                idx_str = m.group("idx") or "1"
                idx = int(idx_str)
                prob = to_prob(m.group("body"))
                _ensure_len(res.answer_conf, idx)
                res.answer_conf[idx-1] = prob
            except (ValueError, IndexError) as e:
                print(f"Error parsing answer confidence with index: {ln} - {str(e)}")
            continue

        # Answer confidence without index
        m = PATTERNS["ans_conf"].match(ln) or PATTERNS["ja_acf"].match(ln)
        if m:
            prob = to_prob(m.group("body"))
            if prob is not None:
                res.answer_conf.append(prob)
            continue

        # Triple with confidence
        m = PATTERNS["triple_w_conf"].match(ln) or PATTERNS["triple_w_vconf"].match(ln) or PATTERNS["triple_w_tf"].match(ln)
        if m:
            try:
                tid_raw = m.group("tid")
                sid, tid = _split_id(tid_raw, current_set)
                ts = res.triple_sets.setdefault(sid, TripleSet())
                _ensure_len(ts.triples, tid)
                _ensure_len(ts.triple_conf, tid)
                ts.triples[tid-1] = m.group("body").strip()
                ts.triple_conf[tid-1] = to_prob(m.group("conf"))
            except (ValueError, IndexError) as e:
                print(f"Error parsing triple: {ln} - {str(e)}")
            continue

        # Triple without confidence
        if m := PATTERNS["triple"].match(ln):
            try:
                tid_raw = m.group("tid")
                sid, tid = _split_id(tid_raw, current_set)
                ts = res.triple_sets.setdefault(sid, TripleSet())
                _ensure_len(ts.triples, tid)
                ts.triples[tid-1] = m.group("body").strip()
            except (ValueError, IndexError) as e:
                print(f"Error parsing triple: {ln} - {str(e)}")
            continue

        # Triple confidence
        if m := PATTERNS["t_conf"].match(ln):
            try:
                tid_raw = m.group("tid") or m.group("tid2")
                if tid_raw:
                    sid, tid = _split_id(tid_raw, current_set)
                    ts = res.triple_sets.setdefault(sid, TripleSet())
                    _ensure_len(ts.triple_conf, tid)
                    ts.triple_conf[tid-1] = to_prob(m.group("body"))
            except (ValueError, IndexError) as e:
                print(f"Error parsing triple confidence: {ln} - {str(e)}")
            continue

        # Overall reasoning confidence to triple
        if m := PATTERNS["think_conf"].match(ln):
            try:
                think_conf = to_prob(m.group("body"))
                if not ts: continue
                if think_conf is not None and len(ts.triples) > 0:
                    for _tid in range(len(ts.triples)):
                        _ensure_len(ts.triple_conf, _tid+1)
                        ts.triple_conf[_tid] = think_conf
            except (ValueError, IndexError) as e:
                print(f"Error parsing think_conf: {ln} - {str(e)}")
            continue

        # Triple confidence (adjusted format)
        if m := PATTERNS["t_conf_adj"].match(ln):
            tid_raw = m.group("tid")
            sid, tid = _split_id(tid_raw, current_set)
            ts = res.triple_sets.setdefault(sid, TripleSet())
            _ensure_len(ts.triple_conf, tid)
            ts.triple_conf[tid-1] = to_prob(m.group("body"))
            continue

    # Ensure all TripleSet triples and triple_conf have the same length
    for ts in res.triple_sets.values():
        max_len = max(len(ts.triples), len(ts.triple_conf))
        _ensure_len(ts.triples, max_len)
        _ensure_len(ts.triple_conf, max_len)

    return res

# Helper functions
def _ensure_len(lst: List, length: int):
    """Extend list to at least 'length' elements (padded with None)"""
    while len(lst) < length:
        lst.append(None)

def _split_id(tid_raw: str, current_set: int) -> Tuple[int, int]:
    """
    Parse triple ID:
    '1'   -> (current_set, 1)
    '1.2' -> (1, 2)
    """
    if not tid_raw:
        return current_set, 1

    if "." in tid_raw:
        parts = tid_raw.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid ID format: {tid_raw}")
        set_id, tri_id = map(int, parts)
        return set_id, tri_id
    return current_set, int(tid_raw)

def extract_answer_and_prob_from_verb_1s_top_1(response, language="en"):
    """
    Extract answer and probability from verb_1s_top_1 format response

    Args:
        response: Model response text
        language: Language ("en" or "ja")

    Returns:
        (answer, probability) tuple
    """
    return parse_output(response)

def extract_answers_and_probs_from_verb_1s_top_k(response, k=2, language="en"):
    """
    Extract answers and probabilities from verb_1s_top_k format response

    Args:
        response: Model response text
        k: Expected number of answers
        language: Language ("en" or "ja")

    Returns:
        (answers, probabilities) tuple
    """
    lines = response.strip().split('\n')

    if language == "ja":
        ans_list, prob_list = parse_lines(lines, "ja_basic")
    else:
        ans_list, prob_list = parse_lines(lines, "en_basic")
    return ans_list, prob_list
