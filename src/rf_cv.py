"""5-fold cross-validation of the surface-feature classifier, to match the protocol of [4].

Random folds over the whole usable set (the same stories appear in train and test folds),
overall accuracy and per query type.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

RAW = Path("data/cladder-v1/cladder-v1-q-balanced.json")
META = Path("data/cladder-v1/cladder-v1-meta-models.json")


def main():
    raw = [r for r in json.load(open(RAW, encoding="utf-8"))
           if isinstance(r.get("reasoning"), dict) and r["reasoning"].get("step1")]
    bg = {m["model_id"]: m["background"] for m in json.load(open(META, encoding="utf-8"))}
    print(f"items={len(raw)}")

    X = [bg[r["meta"]["model_id"]] + " " + r["given_info"] + " " + r["question"] for r in raw]
    y = np.array([1 if r["answer"] == "yes" else 0 for r in raw])
    qt = np.array([r["meta"]["query_type"] for r in raw])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000)),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
        ])
        pipe.fit([X[i] for i in tr], y[tr])
        pred[te] = pipe.predict([X[i] for i in te])

    acc = (pred == y).mean()
    print(f"\noverall 5-fold CV accuracy: {acc:.3f}\n")
    by = defaultdict(list)
    for p, t in zip(pred, y):
        pass
    for p, t, q in zip(pred, y, qt):
        by[q].append(p == t)
    for q in sorted(by):
        a = np.mean(by[q])
        print(f"  {q:20s} n={len(by[q]):5d}  acc={a:.3f}")


if __name__ == "__main__":
    main()
