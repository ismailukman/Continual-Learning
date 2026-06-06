# RSNA Continual Learning

Continual-learning experiments on the RSNA 2023 Abdominal Trauma Detection
dataset. We compare **Baseline (fine-tuning)**, **EWC**, **Experience Replay**,
and **EWC + Replay** on a small CNN, and report average accuracy and forgetting.

## Repository layout

```
.
├── src/                       # experiment code
│   ├── exp_class_incremental.py   # Exp 1: 2-task class-incremental (headline)
│   ├── exp_window_3task_v2.py     # Exp 2: 3-task window domain-incremental
│   ├── exp_window_3task_v3.py     # later window variant (near-chance, kept for ref)
│   ├── exp_improved.py            # mods 1-6 (ResNet-18, λ-sweep, balanced replay,
│   │                              #   herding, LwF, patient agg.) — 2-task, diagnostic
│   ├── exp_improved_v2.py         # 3-window full fine-tune (slice-level; near-chance)
│   ├── exp_mil.py                 # patient-level attention-MIL (best CL result)
│   ├── tune_mil.py                # validation-AUC hyperparameter sweep for exp_mil
│   ├── config.py                  # central experiment configuration + presets
│   ├── utils.py                   # helpers
│   └── quickstart.py              # scaffold (does not load data)
├── notebooks/                 # interactive walkthroughs (local only; empty on GitHub)
├── report/                    # report
│   ├── figures/                   # committed vector figures (PDF) — the only tracked report files
│   └── legacy/                    # earlier plain-text reports (local only; empty on GitHub)
├── logs/                      # run logs / CSVs / matrices (local only; empty on GitHub)
└── data/                      # dataset (local only; empty on GitHub)
```

> **What is on GitHub:** the source code under `src/` and the report's output
> figures under `report/figures/`. Everything else — the dataset, run logs, the
> notebooks, and the report's LaTeX source / compiled PDF / figure-generation
> script — is kept locally and not pushed; `data/`, `logs/`, `notebooks/`, and
> `report/legacy/` appear as empty placeholders.

## Data layout

Expected on disk under `data/`:

```
data/
  RSNA2023ProcessedImages/<patient_id>/<series_id>/<instance>.png
  train.csv               # labels; uses the `any_injury` column
  image_level_labels.csv
```

## Running the experiments

```powershell
conda activate medical_ml

# Smoke test (limits patients/images/iterations)
$env:DEBUG_RUN = "1"; python src/exp_class_incremental.py

# Full run
Remove-Item Env:DEBUG_RUN -ErrorAction SilentlyContinue
python src/exp_class_incremental.py
```

Window-based variants: `python src/exp_window_3task_v2.py`.

### Improved experiment (next iteration)

`src/exp_improved.py` implements the six modifications derived from the report
analysis: (1) pretrained ResNet-18 backbone, (2) EWC λ sweep
{10, 50, 100, 500, 1000}, (3) balanced replay loss, (4) larger buffer + herding
exemplar selection, (5) LwF knowledge distillation, (6) patient/series-level
label aggregation. It is self-contained (no full-run side effects on import).

```powershell
$env:DEBUG_RUN = "1"; python src/exp_improved.py     # fast smoke test (verified)
Remove-Item Env:DEBUG_RUN; python src/exp_improved.py # full run (multi-hour, GPU)
```
