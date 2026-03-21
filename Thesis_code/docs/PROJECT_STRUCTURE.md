# Project Structure

## Current project root

```text
3D-Modeling-of-Microscopic-Structures/
├─ _ThePaper/
├─ BachelorExpose_code/
├─ Thesis_code/
├─ .gitignore
├─ commands.md
└─ README.md
```

## Active code project

All active implementation work happens inside:

```text
Thesis_code/
```

Current structure:

```text
Thesis_code/
├─ .venv/
├─ data/
├─ docs/
│  └─ PROJECT_STATUS.md
├─ results/
├─ src/
├─ tests/
├─ MODEL_B_FEATURES.md
├─ README.md
└─ requirements.txt
```

## Recommended final structure

```text
3D-Modeling-of-Microscopic-Structures/
├─ _ThePaper/
├─ BachelorExpose_code/
├─ Thesis_code/
│  ├─ .venv/
│  ├─ data/
│  │  ├─ datasets/
│  │  └─ test_data/
│  ├─ docs/
│  │  ├─ PROJECT_STATUS.md
│  │  ├─ PROJECT_STRUCTURE.md
│  │  ├─ MODEL_B_FEATURES.md
│  │  └─ commands.md
│  ├─ results/
│  ├─ src/
│  ├─ tests/
│  ├─ README.md
│  └─ requirements.txt
├─ .gitignore
└─ README.md
```

## What each folder is for

### Root folder

* `_ThePaper/` — thesis paper writing and paper-related files
* `BachelorExpose_code/` — old expose-related LaTeX/code material
* `Thesis_code/` — active implementation project
* `.gitignore` — repository ignore rules
* `README.md` — top-level project overview
* `commands.md` — only keep here if it describes the whole umbrella project

### Inside `Thesis_code/`

* `.venv/` — local Python virtual environment
* `data/` — datasets and sample data only
* `docs/` — markdown notes, status files, structure docs, commands, feature notes
* `results/` — generated outputs, previews, checkpoints, logs, plots, training runs
* `src/` — actual source code
* `tests/` — smoke tests and test scripts
* `README.md` — code-project overview and usage
* `requirements.txt` — Python dependencies

## File placement rules

### Keep in root of repository

* `.gitignore`
* `README.md`
* `commands.md` only if it applies to the whole repository

### Keep in `Thesis_code/`

* `.venv`
* `data`
* `docs`
* `results`
* `src`
* `tests`
* `README.md`
* `requirements.txt`

### Move into `Thesis_code/docs/`

* `MODEL_B_FEATURES.md`
* `commands.md` if it is specific to the code project only
* `PROJECT_STATUS.md`
* `PROJECT_STRUCTURE.md`

## Naming conventions

### Datasets

Store all generated datasets in:

```text
Thesis_code/data/datasets/
```

Use names like:

```text
YYYYMMDD_description
```

Examples:

* `20260321_synthetic_B_smoke`
* `20260321_synthetic_B_v2`
* `20260321_modelA_baseline`

### Results

Store all outputs in:

```text
Thesis_code/results/
```

Use names like:

```text
YYYYMMDD_description
```

Examples:

* `20260321_overfit_B_smoke`
* `20260321_train_B_v2`
* `20260321_preview_B`
* `20260321_eval_modelA`

## Project rules going forward

1. Do all active code work from `Thesis_code/`.
2. Run Python commands from inside `Thesis_code/`.
3. Do not create new top-level output folders like `outdir`, `runs2`, `tmp`, or `preview`.
4. Put all generated outputs into `Thesis_code/results/`.
5. Put all generated datasets into `Thesis_code/data/datasets/`.
6. Put all project notes into `Thesis_code/docs/`.
7. Put all code into `Thesis_code/src/`.
8. Put all tests into `Thesis_code/tests/`.

## Command location rule

All normal project commands should be run from:

```text
C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
```

Example:

```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
.\.venv\Scripts\Activate.ps1
python -m src.demo_pipeline --model B --projection max
```

## Immediate cleanup recommendations

### Definitely move

* `Thesis_code/MODEL_B_FEATURES.md` → `Thesis_code/docs/MODEL_B_FEATURES.md`

### Move if code-project specific

* `commands.md` → `Thesis_code/docs/commands.md`

## Status note

This file should be updated whenever:

* a new permanent folder is introduced
* a folder is renamed
* outputs are reorganized
* command execution rules change
* the active project root changes
