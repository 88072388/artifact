"""Surface-feature baseline on the paper's actual splits (comparable to Table 1).

Features: TF-IDF over `given_info + question`. Labels: yes/no.
Train on the 5,781 training questions, evaluate on the four paper test sets.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

W = Path("work_v2")
RAW = Path("data/cladder-v1/cladder-v1-q-balanced.json")


def load_jsonl(name):
    return [json.loads(l) for l in open(W / name, encoding="utf-8") if l.strip()]


def main():
    raw = {str(r["question_id"]): r for r in json.load(open(RAW, encoding="utf-8"))}

    def text_of(i):
        r = raw[str(i)]
        return r["given_info"] + " " + r["question"]

    train = load_jsonl("sft_answer_train.jsonl")
    xtr = [text_of(r["id"]) for r in train]
    ytr = np.array([1 if raw[str(r["id"])]["answer"] == "yes" else 0 for r in train])

    sets = {
        "indist": "eval_indist.jsonl",
        "unseen_struct": "eval_unseen_struct.jsonl",
        "numremoved": "eval_numremoved.jsonl",
        "counterbench": "eval_counterbench.jsonl",
    }
    # CounterBench has no CLadder ids; skip it here (different corpus)
    print(f"train={len(xtr)}  positives={ytr.mean():.3f}\n")

    models = {
        "TFIDF+LogReg": Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "TFIDF+RF": Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000)),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
        ]),
    }

    for name, pipe in models.items():
        pipe.fit(xtr, ytr)
        print(f"== {name}")
        for s, f in sets.items():
            if s == "counterbench":
                continue
            rows = load_jsonl(f)
            xs = [text_of(r["id"]) for r in rows]
            ys = np.array([1 if r["gold_answer"] == "yes" else 0 for r in rows])
            pred = pipe.predict(xs)
            print(f"   {s:14s} n={len(ys):5d}  acc={accuracy_score(ys, pred):.3f}")


if __name__ == "__main__":
    main()
