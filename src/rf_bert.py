"""BERT surface baseline under BOTH protocols, to compare with the reference paper.

(a) 5-fold cross-validation over the usable set  -> comparable to the 87% reported in [4]
(b) story-held-out split                           -> comparable to our fine-tuned models
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

W = Path("work_v2")
RAW = Path("data/cladder-v1/cladder-v1-q-balanced.json")
META = Path("data/cladder-v1/cladder-v1-meta-models.json")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load(name):
    return [json.loads(l) for l in open(W / name, encoding="utf-8") if l.strip()]


@torch.no_grad()
def encode(tok, model, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256,
                return_tensors="pt")
        out.append(model(**b).last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(out)


def probs(text):
    v = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text or "")][:6]
    v = v + [0.0] * (6 - len(v))
    p0, p1 = float(v[0]), float(v[1])
    return [p0, p1, p0 - p1, abs(p0 - p1),
            float(np.mean(v)), float(np.std(v)), min(v), max(v),
            float(max(v) - min(v)), float(len(v))]


def main():
    raw = {str(r["question_id"]): r for r in json.load(open(RAW, encoding="utf-8"))}
    bg = {m["model_id"]: m["background"] for m in json.load(open(META, encoding="utf-8"))}
    usable = [r for r in json.load(open(RAW, encoding="utf-8"))
              if isinstance(r.get("reasoning"), dict) and r["reasoning"].get("step1")]

    def text(r):
        return bg[r["meta"]["model_id"]] + " " + r["given_info"] + " " + r["question"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    print(f"encoding {len(usable)} usable questions ...", flush=True)
    E = encode(tok, model, [text(r) for r in usable])
    P = np.array([probs(r["given_info"]) for r in usable])
    X = np.hstack([E, P])
    y = np.array([1 if r["answer"] == "yes" else 0 for r in usable])

    # (a) 5-fold CV
    print("\n(a) 5-fold cross-validation (comparable to the 87% in [4])")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    print(f"    overall CV accuracy: {accuracy_score(y, pred):.3f}")

    # (b) story-held-out split
    tr_rows = load("sft_answer_train.jsonl")
    idx = {str(r["question_id"]): i for i, r in enumerate(usable)}
    tri = [idx[str(r["id"])] for r in tr_rows]
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0)
    clf.fit(X[tri], y[tri])
    print("\n(b) story-held-out split (comparable to our fine-tuned models)")
    for name, f in [("indist", "eval_indist.jsonl"),
                    ("unseen_struct", "eval_unseen_struct.jsonl"),
                    ("numremoved", "eval_numremoved.jsonl")]:
        rows = load(f)
        ids = [idx[str(r["id"])] for r in rows]
        acc = accuracy_score(y[ids], clf.predict(X[ids]))
        print(f"    {name:14s} n={len(ids):5d}  acc={acc:.3f}")


if __name__ == "__main__":
    main()

