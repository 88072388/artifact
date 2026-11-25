"""Step Fidelity: symbolic, reference-based scoring of each reasoning step.

For every CLadder item, the gold `reasoning` dict gives step0..step5 plus `end`.
Each step is checked by a *type-appropriate* automated checker, so the metric is
reproducible and needs no LLM judge:

    step0  variable naming / roles   -> role-assignment accuracy
    step1  causal graph edges        -> edge-set F1
    step2  estimand (do-notation)    -> token F1 (+ exact match)
    step3  identification formula    -> token F1 (+ exact match)
    step4  numeric substitution      -> value-recall over P(...) assignments
    step5  arithmetic                -> numeric match within tolerance
    end    concluding comparison     -> answer-direction match

Step Fidelity = macro average over the steps that the gold chain actually has.
This matters: some gold chains leave step4 empty (e.g. collider_bias), and those
steps are skipped rather than scored 0.
"""
from __future__ import annotations

import re

from common import STEPS, edge_set, norm, numbers, prob_assignments, token_f1, var_mapping

VALUE_TOL = 0.011  # CLadder probabilities have 2 decimals
NUM_REL_TOL = 0.02


def _sign(text: str) -> int | None:
    """Direction of a concluding comparison: +1 if LHS > RHS, -1 if LHS < RHS."""
    s = (text or "").replace("−", "-")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(>=|>|=<|<=|<|>)\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        if op in (">", ">="):
            return 1 if a >= b else -1
        if op in ("<", "<="):
            return -1 if a <= b else 1
        return 0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*>\s*0", s)
    if m:
        return 1 if float(m.group(1)) > 0 else -1
    return None


def _answer_of(text: str):
    hits = re.findall(r"\b(yes|no)\b", (text or "").lower())
    return hits[-1] if hits else None


def _score_step0(gold: dict, pred: dict) -> float | None:
    g = var_mapping(gold.get("step0", ""))
    p = var_mapping(pred.get("step0", ""))
    if not g:
        return None
    if not p:
        return 0.0
    hits = 0.0
    for var, desc in g.items():
        if var in p:
            hits += 1.0 if token_f1(desc, p[var]) >= 0.5 else 0.0
    return hits / len(g)


def _score_step1(gold: dict, pred: dict) -> float | None:
    g = edge_set(gold.get("step1", ""))
    p = edge_set(pred.get("step1", ""))
    if not g:
        return None
    if not p:
        return 0.0
    inter = len(g & p)
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(g)
    return 2 * precision * recall / (precision + recall)


def _score_formula(gold_text: str, pred_text: str):
    if not (gold_text or "").strip():
        return None, None
    if not (pred_text or "").strip():
        return 0.0, False
    return token_f1(gold_text, pred_text), norm(gold_text) == norm(pred_text)


def _score_step4(gold: dict, pred: dict) -> float | None:
    g = prob_assignments(gold.get("step4", ""))
    if not g:
        return None
    p = prob_assignments(pred.get("step4", ""))
    if not p:
        return 0.0
    ok = sum(1 for k, v in g.items() if k in p and abs(p[k] - v) <= VALUE_TOL)
    missed = len(g) - ok
    extra = max(0, len(p) - ok)
    precision = ok / max(1, len(p))
    recall = ok / len(g)
    f1 = 0.0 if ok == 0 else 2 * precision * recall / (precision + recall)
    return f1


def _score_step5(gold: dict, pred: dict) -> float | None:
    g = numbers(gold.get("step5", ""))
    if not g:
        return None
    p = numbers(pred.get("step5", ""))
    if not p:
        return 0.0
    target = g[-1]
    tol = max(NUM_REL_TOL, 0.02 * abs(target))
    return 1.0 if any(abs(x - target) <= tol for x in p) else 0.0


def _score_end(gold: dict, pred: dict) -> float | None:
    g_text = gold.get("end", "") or ""
    if not g_text.strip():
        return None
    g_sign = _sign(g_text)
    if g_sign is None:
        ga, pa = _answer_of(g_text), _answer_of(pred.get("conclusion", ""))
        if ga is None:
            return None
        return 1.0 if ga == pa else 0.0
    p_sign = _sign(pred.get("conclusion", ""))
    if p_sign is None:
        return 0.0
    return 1.0 if p_sign == g_sign else 0.0


def score_steps(gold: dict, pred: dict) -> dict:
    """gold/pred: dicts with step0..step5, conclusion. Returns per-step and aggregate scores."""
    per = {}
    per["step0"] = _score_step0(gold, pred)
    per["step1"] = _score_step1(gold, pred)
    per["step2"], ex2 = _score_formula(gold.get("step2", ""), pred.get("step2", ""))
    per["step3"], ex3 = _score_formula(gold.get("step3", ""), pred.get("step3", ""))
    per["step4"] = _score_step4(gold, pred)
    per["step5"] = _score_step5(gold, pred)
    per["end"] = _score_end(gold, pred)

    graded = {k: v for k, v in per.items() if v is not None}
    fidelity = sum(graded.values()) / len(graded) if graded else 0.0

    present = sum(1 for k in STEPS if (pred.get(k) or "").strip())
    exact = sum(
        1
        for k, e in (("step2", ex2), ("step3", ex3))
        if e and (gold.get(k) or "").strip()
    )
    return {
        **{f"s_{k}": (round(v, 4) if v is not None else None) for k, v in per.items()},
        "fidelity": round(fidelity, 4),
        "n_graded_steps": len(graded),
        "format_compliance": round(present / len(STEPS), 4),
        "formula_exact": exact,
    }
