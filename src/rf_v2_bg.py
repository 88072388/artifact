"""Surface-feature baseline that also sees the background paragraph (the fairer comparison).

Same split and labels as rf_v2.py; the only difference is that the feature text includes the
story background, which is what the model actually sees.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

W = Path("work_v2")
RAW = Path("data/cladder-v1/cladder-v1-q-balanced.json")
META = Path("data/cladder-v1/cladder-v1-meta-models.json")


def load(name):
    return [json.loads(l) for l in open(W / name, encoding="utf-8") if l.strip()]


def main():
    raw = {str(r["question_id"]): r for r in json.load(open(RAW, encoding="utf-8"))}
    bg = {m["model_id"]: m["background"] for m in json.load(open(META, encoding="utf-8"))}

    def text(i):
        r = raw[str(i)]
        return bg[r["meta"]["model_id"]] + " " + r["given_info"] + " " + r["question"]

    train = load("sft_answer_train.jsonl")
    X = [text(r["id"]) for r in train]
    y = np.array([1 if raw[str(r["id"])]["answer"] == "yes" else 0 for r in train])

    pipe = Pipeline([
        ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000)),
        ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
    ])
    pipe.fit(X, y)
    print(f"train={len(X)}  positives={y.mean():.3f}\n")

    for name, f in [("indist", "eval_indist.jsonl"),
                    ("unseen_struct", "eval_unseen_struct.jsonl"),
                    ("numremoved", "eval_numremoved.jsonl")]:
        rows = load(f)
        xs = [text(r["id"]) for r in rows]
        ys = np.array([1 if r["gold_answer"] == "yes" else 0 for r in rows])
        print(f"  {name:14s} n={len(ys):5d}  acc={accuracy_score(ys, pipe.predict(xs)):.3f}")


if __name__ == "__main__":
    main()
