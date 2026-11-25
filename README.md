# Fidelity measurement for fine-tuned causal reasoners (anonymous artefact)

Everything needed to rebuild the numbers in the paper from the released
summaries, or to retrain from scratch on CLadder.

## Layout

    src/          training, evaluation, probing, scoring
    scripts/      end-to-end shell drivers (order: 01 -> 08)
    data/         item-level split manifest (usable_ids.jsonl)
    annotations/  seventy-step checker-vs-human annotation export
    results/      per-run summary JSON (accuracy, F(q), step scores)
    RUN.md        command-by-command reproduction
    requirements.txt  pinned environment

## What each table needs

| Paper table | Files | Command |
|---|---|---|
| Table 1 (in-distribution) | results/*-indist.summary.json | `python src/make_tables_v2.py` |
| Table 2 (out-of-distribution) | results/*-{unseen,numremoved,counterbench}.summary.json | same |
| Table 3 (intervention, 328 items) | results/probes_*-p328.json | `python src/score_any.py <results dir> <results7b> <results_llama>` |
| Figure 3 (per-step fidelity) | results/*-process-s*-indist.summary.json | same |

`probes_*.json` store a gold-anchored `flip_rate`; it is **not** the number in
the paper. Table 3 is model-anchored and must be recomputed with
`src/score_any.py`, which compares an answer against the model's own `keep`
answer. Reading the field directly gives 0.000 for the answer arm.

## Data

CLadder (NeurIPS 2023, `causalNLP/causal-reasoning`) is downloaded by
`scripts/v2_01_build.sh`. We drop 1,580 items whose reasoning trace is null and
229 items with a negligible effect size, leaving 8,303 usable items.
`data/usable_ids.jsonl` lists every item we used with its split, so the exact
train/val/test partition can be reconstructed without redistributing CLadder.

CounterBench (AAAI-26) is used as an out-of-distribution set only.

## Environment

    pip install -r requirements.txt

Four-bit QLoRA on a single 24 GB card. `bitsandbytes` is required.

## Licence

Code: MIT. Derived data files are released under the terms of the upstream
benchmarks; CLadder and CounterBench keep their own licences.
