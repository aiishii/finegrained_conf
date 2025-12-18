#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gzip
import csv
import random
from typing import List, Dict, Tuple, Optional, Any
import pathlib
import pandas as pd

class Dataset:
    """Dataset container class"""

    def __init__(self, name: str, questions: List[str], answers: List[str], derivations=None, qid:List[str]=None, qtype:List[str]=None):
        """
        Initialize the dataset

        Args:
            name: Dataset name
            questions: List of questions
            answers: List of answers (must be same length as questions)
        """
        assert len(questions) == len(answers), "Number of questions and answers must match"
        self.name = name
        self.questions = questions
        self.answers = answers
        self.derivations = derivations
        self.qid = qid
        self.qtype = qtype

    def __len__(self) -> int:
        """Return the length of the dataset"""
        return len(self.questions)

    def get_sample(self, idx: int) -> Dict[str, str]:
        """
        Get sample at specified index

        Args:
            idx: Sample index

        Returns:
            Dictionary containing question and answer
        """
        if self.qid and self.qtype and self.derivations:
            return {
                "qid": self.qid[idx],
                "qtype": self.qtype[idx],
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "derivations": self.derivations[idx]
            }
        if self.derivations:
            return {
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "derivations": self.derivations[idx]
            }
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx]
        }


    def get_random_samples(self, num_samples: int) -> 'Dataset':
        """
        Get specified number of random samples

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            New Dataset object containing random samples
        """
        if num_samples >= len(self):
            return self

        indices = random.sample(range(len(self)), num_samples)
        questions = [self.questions[i] for i in indices]
        answers = [self.answers[i] for i in indices]
        if self.derivations:
            derivations = [self.derivations[i] for i in indices]
            return Dataset(self.name, questions, answers, derivations=derivations)

        return Dataset(self.name, questions, answers)

def load_trivia_qa(filepath: str, split: str = "validation", max_samples: int = None) -> Dataset:
    """
    Load TriviaQA dataset

    Args:
        filepath: Path to data file
        split: Data split to use ("train", "validation", "test")
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object
    """
    questions = []
    answers = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data['Data']:
            question = item['Question']
            answer = item['Answer']['Value']

            questions.append(question)
            answers.append(answer)

            if max_samples and len(questions) >= max_samples:
                break

    except Exception as e:
        print(f"Error loading TriviaQA: {e}")
        questions = ["What was the name of the first artificial Earth satellite?"] * 10
        answers = ["Sputnik 1"] * 10

        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]

    return Dataset("trivia_qa", questions, answers)

def load_sci_q(filepath: str, split: str = "validation", max_samples: int = None) -> Dataset:
    """
    Load SciQ dataset

    Args:
        filepath: Path to data file
        split: Data split to use ("train", "validation", "test")
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object
    """
    questions = []
    answers = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                question = item['question']
                answer = item['correct_answer']

                questions.append(question)
                answers.append(answer)

                if max_samples and len(questions) >= max_samples:
                    break

    except Exception as e:
        print(f"Error loading SciQ: {e}")
        questions = ["What is the closest planet to the Sun?"] * 10
        answers = ["Mercury"] * 10

        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]

    return Dataset("sci_q", questions, answers)

def load_truthful_qa(filepath: str, split: str = "validation", max_samples: int = None) -> Dataset:
    """
    Load TruthfulQA dataset

    Args:
        filepath: Path to data file
        split: Data split to use ("train", "validation", "test")
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object
    """
    questions = []
    answers = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = row['Question']
                answer = row['Best Answer']

                questions.append(question)
                answers.append(answer)

                if max_samples and len(questions) >= max_samples:
                    break

    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
        questions = ["Does reading in dim light damage your eyes?"] * 10
        answers = ["No"] * 10

        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]

    return Dataset("truthful_qa", questions, answers)

def load_jemhop_qa(filepath: str, split: str = "dev", max_samples: int = None) -> Dataset:
    """
    Load JEMHopQA dataset

    Args:
        filepath: Path to data file
        split: Data split to use ("train", "dev")
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object
    """
    qids = []
    qtypes = []
    questions = []
    answers = []
    derivations = []

    try:
        p_filepath = pathlib.Path(filepath)
        df = pd.read_json(p_filepath)

        for idx, r in df.iterrows():
            question = r.question
            answer = r.answer
            qid = r.qid
            qtype = r.type

            deriv_list = []
            for deriv in r.derivations:
                for multi_value in deriv[2]:
                    deriv_list.append("（" + "，".join([deriv[0], deriv[1], multi_value]) + "）")
            gold_derivations = ';'.join(deriv_list)

            questions.append(question)
            answers.append(answer)
            derivations.append(gold_derivations)
            qids.append(qid)
            qtypes.append(qtype)

            if max_samples and len(questions) >= max_samples:
                break

    except Exception as e:
        print(f"Error loading JEMHopQA: {e}")
        questions = ["『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？"] * 10
        answers = ["あまちゃん"] * 10

        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]

    return Dataset("jemhop_qa", questions, answers, derivations, qid=qids, qtype=qtypes)

def load_hotpot_qa(filepath: str = 'data/hotpot_qa/r4c_dev_select200_20240119_165154.tsv', split: str = "dev", max_samples: int = None) -> Dataset:
    """
    JEMHopQAデータセットを読み込む
    
    Args:
        filepath: データファイルのパス
        split: 使用するデータ分割（"train", "dev"）
        max_samples: 読み込むサンプルの最大数
        
    Returns:
        Datasetオブジェクト
    """
    questions = []
    answers = []
    derivations = []
    
    try:
        # JEMHopQAはJSON形式
        p_filepath = pathlib.Path(filepath)
        df = pd.read_table(p_filepath)

        for idx, r in df.iterrows():
            question = r.question
            answer = r.answer
            gold_derivations = r.derivations

            questions.append(question)
            answers.append(answer)
            derivations.append(gold_derivations)

            if max_samples and len(questions) >= max_samples:
                break
   
    except Exception as e:
        print(f"HotPotQAの読み込み中にエラーが発生しました: {e}")
        # 論文の実装を再現するためのダミーデータ
        
        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]
    
    return Dataset("hotpot_qa", questions, answers, derivations)

def load_2wiki_qa(filepath: str = 'data/2wiki_qa/dev_select1000.json', split: str = "dev", max_samples: int = None) -> Dataset:
    """
    2WikiMultihopQAデータセットを読み込む
    
    Args:
        filepath: データファイルのパス
        split: 使用するデータ分割（"train", "dev"）
        max_samples: 読み込むサンプルの最大数
        
    Returns:
        Datasetオブジェクト
    """
    qids = []
    qtypes = []
    questions = []
    answers = []
    derivations = []
    
    # try:
    # JEMHopQAはJSON形式
    p_filepath = pathlib.Path(filepath)
    df = pd.read_json(p_filepath)

    for idx, r in df.iterrows():
        question = r.question
        answer = r.answer
        qid = r.qid
        qtype = r.type

        deriv_list = []
        for deriv in r.derivations:
            deriv_list.append(f'("{deriv[0]}", "{deriv[1]}", "{deriv[2]}")')
        gold_derivations = ';'.join(deriv_list)

        questions.append(question)
        answers.append(answer)
        derivations.append(gold_derivations)
        qids.append(qid)
        qtypes.append(qtype)

        if max_samples and len(questions) >= max_samples:
            break
   
    return Dataset("2wiki_qa", questions, answers, derivations, qid=qids, qtype=qtypes)

def load_dataset(dataset_name: str, dataset_path: str = None, split: str = "validation", max_samples: int = None) -> Dataset:
    default_paths = {
        "trivia_qa": "data/trivia_qa/qa/web-{}.json".format(split),
        "sci_q": "data/sci_q/{}.jsonl".format(split),
        "truthful_qa": "data/truthful_qa/truthful_qa.csv",
        "jemhop_qa": "data/jemhop_qa/{}_ver1.2.json".format(split),
        "hotpot_qa": "data/hotpot_qa/r4c_dev_select200_20240119_165154.tsv",
        "2wiki_qa": "data/2wiki_qa/dev_select1000.json"
    }
    
    if dataset_path is None:
        dataset_path = default_paths.get(dataset_name, "")

    if not os.path.exists(dataset_path):
        print(f"警告: データセットファイル {dataset_path} が見つかりません。サンプルデータを使用します。")
        derivations = []
        if dataset_name == "trivia_qa":
            questions = ["What was the name of the first artificial Earth satellite?",
                        "Who wrote 'Pride and Prejudice'?",
                        "What is the capital of Japan?",
                        "What is the largest planet in our solar system?",
                        "In which year did World War II end?"]
            answers = ["Sputnik 1", 
                      "Jane Austen", 
                      "Tokyo", 
                      "Jupiter", 
                      "1945"]
        elif dataset_name == "sci_q":
            questions = ["What is the closest planet to the Sun?",
                        "What is the chemical symbol for gold?",
                        "What is the hardest natural substance on Earth?",
                        "What is the largest organ in the human body?",
                        "What is the most abundant gas in Earth's atmosphere?"]
            answers = ["Mercury", 
                      "Au", 
                      "Diamond", 
                      "Skin", 
                      "Nitrogen"]
        elif dataset_name == "truthful_qa":
            questions = ["Does reading in dim light damage your eyes?",
                        "Is it dangerous to wake a sleepwalker?",
                        "Does cracking your knuckles cause arthritis?",
                        "Does sugar make children hyperactive?",
                        "Do we only use 10% of our brains?"]
            answers = ["No", 
                      "No", 
                      "No", 
                      "No", 
                      "No"]
        elif dataset_name == "jemhop_qa":
            questions = ["『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？",
                        "清原果耶と森田成一で活動開始が早かったのはどちらですか？",
                        "孝明天皇が生涯過ごした都に以前の都から遷都があった年は？",
                        "『ぼくらが旅に出る理由』と『ボーン・トゥ・ラヴ・ユー』はどちらも小沢健二のシングルですか？",
                        "無名塾の主宰者の誕生日の年月日は？"]
            answers = ["あまちゃん",
                        "森田成一",
                        "794年",
                        "NO",
                        "1932年12月13日"]
            derivations = ["（仮面ライダー電王，放送回数，49）;（あまちゃん，放送回数，156）",
                        "（清原果耶，活動開始時，2015年）;（森田成一，活動開始時，2001年）",
                        "（孝明天皇，生涯を過ごした都，平安京）;（平安京，遷都された年，794年）",
                        "（ぼくらが旅に出る理由，企画・制作，小沢健二）;（ボーン・トゥ・ラヴ・ユー，企画・制作，フレディ・マーキュリー）",
                        "（無名塾，主宰者，仲代達矢）;（仲代達矢，生年月日，1932年12月13日）"]
        elif dataset_name == "hotpot_qa":
            questions = ["『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？",
                        "清原果耶と森田成一で活動開始が早かったのはどちらですか？",
                        "孝明天皇が生涯過ごした都に以前の都から遷都があった年は？",
                        "『ぼくらが旅に出る理由』と『ボーン・トゥ・ラヴ・ユー』はどちらも小沢健二のシングルですか？",
                        "無名塾の主宰者の誕生日の年月日は？"]
            answers = ["あまちゃん",
                        "森田成一",
                        "794年",
                        "NO",
                        "1932年12月13日"]
            derivations = ["（仮面ライダー電王，放送回数，49）;（あまちゃん，放送回数，156）",
                        "（清原果耶，活動開始時，2015年）;（森田成一，活動開始時，2001年）",
                        "（孝明天皇，生涯を過ごした都，平安京）;（平安京，遷都された年，794年）",
                        "（ぼくらが旅に出る理由，企画・制作，小沢健二）;（ボーン・トゥ・ラヴ・ユー，企画・制作，フレディ・マーキュリー）",
                        "（無名塾，主宰者，仲代達矢）;（仲代達矢，生年月日，1932年12月13日）"]
        elif dataset_name == "2wiki_qa":
            questions = ["『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？",
                        "清原果耶と森田成一で活動開始が早かったのはどちらですか？",
                        "孝明天皇が生涯過ごした都に以前の都から遷都があった年は？",
                        "『ぼくらが旅に出る理由』と『ボーン・トゥ・ラヴ・ユー』はどちらも小沢健二のシングルですか？",
                        "無名塾の主宰者の誕生日の年月日は？"]
            answers = ["あまちゃん",
                        "森田成一",
                        "794年",
                        "NO",
                        "1932年12月13日"]
            derivations = ["（仮面ライダー電王，放送回数，49）;（あまちゃん，放送回数，156）",
                        "（清原果耶，活動開始時，2015年）;（森田成一，活動開始時，2001年）",
                        "（孝明天皇，生涯を過ごした都，平安京）;（平安京，遷都された年，794年）",
                        "（ぼくらが旅に出る理由，企画・制作，小沢健二）;（ボーン・トゥ・ラヴ・ユー，企画・制作，フレディ・マーキュリー）",
                        "（無名塾，主宰者，仲代達矢）;（仲代達矢，生年月日，1932年12月13日）"]
        
        else:
            raise ValueError(f"不明なデータセット: {dataset_name}")
        
        if max_samples:
            questions = (questions * (max_samples // len(questions) + 1))[:max_samples]
            answers = (answers * (max_samples // len(answers) + 1))[:max_samples]
            if derivations:
                derivations = (derivations * (max_samples // len(derivations) + 1))[:max_samples]
        if derivations:
            return Dataset(dataset_name, questions, answers, derivations=derivations)
        return Dataset(dataset_name, questions, answers)
    
    if dataset_name == "trivia_qa":
        return load_trivia_qa(dataset_path, split, max_samples)
    elif dataset_name == "sci_q":
        return load_sci_q(dataset_path, split, max_samples)
    elif dataset_name == "truthful_qa":
        return load_truthful_qa(dataset_path, split, max_samples)
    elif dataset_name == "jemhop_qa":
        return load_jemhop_qa(dataset_path, split, max_samples)
    elif dataset_name == "hotpot_qa":
        return load_hotpot_qa(dataset_path, split, max_samples)
    elif dataset_name == "2wiki_qa":
        return load_2wiki_qa(dataset_path, split, max_samples)
    else:
        raise ValueError(f"不明なデータセット: {dataset_name}")

if __name__ == "__main__":
    # 使用例
    dataset_list = ["trivia_qa", "sci_q", "truthful_qa"]
    dataset_list = ["jemhop_qa"]
    dataset_list = ["hotpot_qa"]
    max_samples = 5
    for dataset_name in dataset_list:
        dataset = load_dataset(dataset_name, split="dev", max_samples=max_samples)
        print(f"{dataset.name}: {len(dataset)} サンプル")
        for i in range(min(max_samples, len(dataset))):
            sample = dataset.get_sample(i)
            # print(sample)
            print(f"  質問: {sample['question']}")
            print(f"  回答: {sample['answer']}")
            if 'derivations' in sample:
                print(f"  導出: {sample['derivations']}")
            print()