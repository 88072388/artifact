"""Shared utilities: data loading, prompt building, output parsing, normalization."""
from __future__ import annotations

import json
import re
from pathlib import Path

STEPS = ["step0", "step1", "step2", "step3", "step4", "step5"]

# ---------------------------------------------------------------- data loading


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_backgrounds(meta_models_path: str | Path) -> dict:
    """model_id -> background text."""
    models = read_json(meta_models_path)
    return {m["model_id"]: m["background"] for m in models}


def has_steps(rec) -> bool:
    return isinstance(rec.get("reasoning"), dict) and bool(rec["reasoning"].get("step1"))


# ---------------------------------------------------------------- prompting

ANSWER_ONLY_INSTRUCTION = (
    "Answer the question with a single word, yes or no.\n"
    'Reply exactly in the format: "Answer: yes" or "Answer: no".'
)

PROCESS_INSTRUCTION = """\
Answer the question by writing the causal reasoning in exactly the format below, \
then give the final answer. Use the variable names X, Y, V1, V2, V3.

Step 0: name each variable and its role.
Step 1: write the causal graph as directed edges, e.g. X->Y,V2->Y
Step 2: write the quantity to be estimated in do-notation.
Step 3: write how to compute it from the observed probabilities.
Step 4: plug in the observed numbers.
Step 5: carry out the arithmetic and report one number.
Conclusion: write the comparison that decides the answer.
Answer: yes or no"""


def build_prompt(rec: dict, background: str, mode: str = "answer") -> str:
    """mode: 'answer' (answer-only) or 'process' (full reasoning chain)."""
    instr = ANSWER_ONLY_INSTRUCTION if mode == "answer" else PROCESS_INSTRUCTION
    return (
        f"{background}\n\n{rec['given_info']}\n\n{rec['question']}\n\n{instr}"
    )


def build_target(rec: dict, mode: str) -> str:
    if mode == "answer":
        return f"Answer: {rec['answer']}"
    r = rec["reasoning"]
    lines = []
    for i, k in enumerate(STEPS):
        txt = (r.get(k) or "").strip()
        lines.append(f"Step {i}: {txt}")
    lines.append(f"Conclusion: {(r.get('end') or '').strip()}")
    lines.append(f"Answer: {rec['answer']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- parsing

_MD = re.compile(r"[*_`]")
_STEP_RE = re.compile(r"step\s*([0-5])\s*[:：]\s*(.*)", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^\s*(step\s*[0-5]\s*[:：]|conclusion\s*[:：]|answer\s*[:：]|solution\s*[:：])",
    re.IGNORECASE,
)


def _clean(line: str) -> str:
    return _MD.sub("", line).strip()


def parse_steps(text: str) -> dict:
    """Parse model output into {'step0'..'step5', 'conclusion': str, 'answer': str|None}."""
    out = {k: "" for k in STEPS}
    out["conclusion"] = ""
    out["answer"] = None
    if not text:
        return out

    current = None
    buf: list[str] = []

    def flush():
        nonlocal buf, current
        if current is not None:
            joined = " ".join(b for b in buf if b).strip()
            if current in out and current != "answer":
                out[current] = (out[current] + " " + joined).strip()
        buf = []

    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        m = _STEP_RE.match(line)
        if m:
            flush()
            current = f"step{m.group(1)}"
            buf = [m.group(2)]
            continue
        if re.match(r"^conclusion\s*[:：]", line, re.IGNORECASE):
            flush()
            current = "conclusion"
            buf = [re.sub(r"^conclusion\s*[:：]", "", line, flags=re.IGNORECASE)]
            continue
        if re.match(r"^answer\s*[:：]", line, re.IGNORECASE):
            flush()
            current = "answer"
            continue
        if current is not None:
            buf.append(line)
    flush()

    out["answer"] = extract_answer(text)
    if not out["conclusion"]:
        out["conclusion"] = extract_conclusion(text)
    return out


_ANS_RE = re.compile(r"answer\s*(?:is|:)?\s*[:：]?\s*(yes|no)\b", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    m = _ANS_RE.findall(text)
    if m:
        return m[-1].lower()
    # fallback: last standalone yes/no near the end of the output
    tail = text[-120:].lower()
    hits = re.findall(r"\b(yes|no)\b", tail)
    return hits[-1] if hits else None


_CONC_RE = re.compile(r"conclusion\s*[:：]\s*(.+)", re.IGNORECASE)


def extract_conclusion(text: str) -> str:
    m = _CONC_RE.search(text or "")
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------- normalization

_SPACE = re.compile(r"\s+")


def norm(s: str) -> str:
    """Aggressive normalization for formula-ish comparison."""
    s = (s or "").lower()
    s = _MD.sub("", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("−", "-").replace("×", "*").replace("·", "*")
    s = s.replace("→", "->")
    s = _SPACE.sub("", s)
    s = s.replace("**", "")
    return s


_TOK = re.compile(r"[a-z0-9]+|->|\+|-|\*|/|\(|\)|\[|\]|\{|\}|=|,|\||\.")


def token_f1(a: str, b: str) -> float:
    ta, tb = _TOK.findall(norm(a).lower()), _TOK.findall(norm(b).lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


_EDGE = re.compile(r"([A-Za-z]\w*)\s*-{1,2}>\s*([A-Za-z]\w*)")


def edge_set(s: str) -> set[tuple[str, str]]:
    s = (s or "").replace("→", "->")
    return {(a.upper(), b.upper()) for a, b in _EDGE.findall(s) if a.upper() != b.upper()}


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(s: str) -> list[float]:
    return [float(x) for x in _NUM.findall(s or "")]


_ASSIGN = re.compile(r"([A-Za-z]\w*)\s*=\s*([^0-9=,;|]{1,60}?)(?=[,;|]|$)")


def var_mapping(s: str) -> dict[str, str]:
    """'Let X = husband; V2 = wife.' -> {'X': 'husband', 'V2': 'wife'}"""
    out = {}
    for var, desc in _ASSIGN.findall(s or ""):
        d = _SPACE.sub(" ", desc).strip(" .'\"")
        if d and not d.replace(".", "").isdigit():
            out[var.upper()] = d.lower()
    return out


_PROB = re.compile(r"([^=\n]{2,80}?)=\s*(-?\d+(?:\.\d+)?)")


def prob_assignments(s: str) -> dict[str, float]:
    out = {}
    for lhs, val in _PROB.findall(s or ""):
        key = norm(lhs)
        if key:
            out[key] = float(val)
    return out
