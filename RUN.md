# Reproducibility package

Step-level fidelity and intervention probes for LoRA-tuned causal reasoning.

## What is in here

```
src/        training, evaluation, probe and table scripts
scripts/    the exact command sequence we ran
results/    every summary file behind the paper's tables (107 files)
RUN.md      this file, including the numbers you should reproduce
```

Not included: CLadder and CounterBench (both public, MIT) and the
trained adapters (~17 MB each). Download the datasets from their
official links; adapters are available on request.

## Environment

Tested on a single RTX 4090 (24 GB), Linux, Python 3.12,
`torch 2.5.1+cu124`, `transformers 4.46.3`, `vllm 0.6.6`, `numpy 1.26.4`.

numpy must stay below 2.0 (vLLM 0.6.6 requirement). `requirements.txt`
pins the exact set; do not upgrade it.

## Reproduce

```bash
# 1. data  (CLadder + CounterBench must be present under data/)
python src/build_v2.py --data-dir data/cladder-v1 --out-dir work_v2
python src/build_counterbench.py --data-dir data/counterbench --out work_v2/eval_counterbench.jsonl

# 2. train 21 LoRA adapters (3 models x 2 arms x 3 seeds, plus 3 scrambled)
bash scripts/v2_03_train.sh

# 3. evaluate: 4 test sets + intervention probes
bash scripts/v2_04_eval.sh

# 3b. zero-shot baselines (no training, just inference with the base models)
bash scripts/v2_05_baseline.sh

# 3c. intervention probes on the 328-item set (reported in the paper)
python src/build_probes_p328.py
bash scripts/v2_06_probes328.sh

# 4. tables
python src/make_tables_v2.py
```

`v2_03_train.sh` and `v2_04_eval.sh` both skip work whose output already
exists, so they can be re-run after an interruption.

## Surface baselines (not needed for the main results)

`ash
python src/rf_v2.py        # TF-IDF + RF, story-held-out split
python src/rf_strong.py    # + numeric, query-type and graph features
python src/rf_cv.py        # 5-fold CV over all 8,532 annotated questions
python src/rf_bert.py      # BERT embeddings + explicit probabilities (needs torch/transformers)
`


f_bert.py expects sentence-transformers/all-MiniLM-L6-v2; point the MODEL
constant at a local copy if the machine is offline. The paper reports 0.533
(story-held-out) and 0.605 (cross-validated) from these scripts.

These scripts additionally require torch and transformers; the main pipeline does not, so they are absent from requirements.txt.

## Expected numbers

If your run is correct you should see these (mean over 3 seeds):

| Check | Expected |
|---|---|
| usable questions after filtering | 8,303 |
| train / val / in-dist test / unseen-structure | 5,781 / 528 / 1,089 / 905 |
| number-removed probe / CounterBench | 1,499 / 1,200 |
| intervention probes | 157 items x 3 conditions = 471 rows |
| adapters produced | 21 |
| summary files produced | 84 (21 x 4) + 12 zero-shot |

In-distribution accuracy:

| Model | zero-shot | A-FT | P-FT |
|---|---|---|---|
| 0.5B | 0.479 | 0.543 | 0.496 |
| 1.5B | 0.508 | 0.842 | 0.644 |
| 3B | 0.483 | 0.855 | 0.730 |

## Known data quirks

Two properties of CLadder that we handle explicitly:

1. **`backadj` has no `reasoning` field** (1,580 of 10,112 items). These are
   dropped; both supervision arms see the identical 8,303 items.
2. **The step-5 arithmetic uses rounded inputs**, so re-evaluating the
   expression differs from the stored result by more than 0.01 in 3,822 of
   6,785 parseable items. We keep the stored result (equal to the ground
   truth) and compare numbers with a 0.02 tolerance.

## Split hygiene

Splits are grouped by narrative story. The eight in-distribution test
stories never occur in training; the `diamond` graph is absent from
training entirely. No test prompt is an exact duplicate of a training
prompt, and all intervention probes are drawn from the test sets.

Note that 8 of the 13 stories in the unseen-structure split also appear in
training with other graphs, so that split isolates structural
generalisation rather than domain shift.
