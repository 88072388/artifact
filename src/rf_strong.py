"""Stronger surface baseline: lexical + numeric + query-type + graph features."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

W = Path("work_v2")
RAW = Path("data/cladder-v1/cladder-v1-q-balanced.json")
META = Path("data/cladder-v1/cladder-v1-meta-models.json")
NUMCOLS = [f"n{i}" for i in range(6)]


def load(name):
    return [json.loads(l) for l in open(W / name, encoding="utf-8") if l.strip()]


def nums(text):
    v = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text or "")][:6]
    if not v:
        v = [0.0]
    v = v + [0.0] * (6 - len(v))
    return [float(np.mean(v)), float(np.std(v)), min(v), max(v),
            float(max(v) - min(v)), float(len(v))]


def main():
    raw = {str(r["question_id"]): r for r in json.load(open(RAW, encoding="utf-8"))}
    bg = {m["model_id"]: m["background"] for m in json.load(open(META, encoding="utf-8"))}

    def frame(ids, labels=None):
        recs = []
        for i in ids:
            r = raw[str(i)]
            d = {"txt": bg[r["meta"]["model_id"]] + " " + r["given_info"] + " " + r["question"],
                 "qt": r["meta"]["query_type"], "gr": r["meta"]["graph_id"]}
            d.update(dict(zip(NUMCOLS, nums(r["given_info"]))))
            recs.append(d)
        df = pd.DataFrame(recs)
        return df, labels

    tr = load("sft_answer_train.jsonl")
    Xtr, _ = frame([r["id"] for r in tr])
    ytr = np.array([1 if raw[str(r["id"])]["answer"] == "yes" else 0 for r in tr])

    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000), "txt"),
        ("num", "passthrough", NUMCOLS),
        ("qt", OneHotEncoder(handle_unknown="ignore"), ["qt"]),
        ("gr", OneHotEncoder(handle_unknown="ignore"), ["gr"]),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0))])
    pipe.fit(Xtr, ytr)
    print(f"train={len(Xtr)}")

    for name, f in [("indist", "eval_indist.jsonl"),
                    ("unseen_struct", "eval_unseen_struct.jsonl"),
                    ("numremoved", "eval_numremoved.jsonl")]:
        rows = load(f)
        X, _ = frame([r["id"] for r in rows])
        y = np.array([1 if r["gold_answer"] == "yes" else 0 for r in rows])
        print(f"  {name:14s} n={len(y):5d}  acc={accuracy_score(y, pipe.predict(X)):.3f}")


if __name__ == "__main__":
    main()
