# demo_pipeline.py

## Purpose

This file is a simple demonstration script for Phase 1 of the synthetic microscopy pipeline.

Its purpose is to generate one example synthetic sample and visualize the key outputs side by side.

It is mainly used for quick inspection and debugging, not for full dataset generation or model training.

The script shows:

- the raw projected mask,
- the visible refined mask,
- the clean rendered image,
- the noisy final image.

This makes it very useful for checking whether the current generation and masking settings produce sensible results.

---

## Main Dependencies

This file depends mainly on:

- `generate_single_example(...)` from `dataset_generator.py`
- `_mask_to_2d(...)` from `dataset_generator.py`
- `matplotlib.pyplot`

So this file does not implement the actual synthetic generation logic itself.  
Instead, it calls the dataset-generation pipeline and visualizes the results.

---

## Inputs

### Main function inputs

The `main(...)` function accepts many parameters, most of which are passed directly to `generate_single_example(...)`.

These can be grouped into the following categories.

---

### 1. Structure / volume parameters

For Model A:

- `n_tubes`
- `tube_radius`

For Model B:

- `n_seeds`
- `step_length`
- `n_steps`
- `branch_prob`
- `branch_steps`
- `radius_min`
- `radius_max`
- `persistence`
- `z_scale`
- `seed_spread`

---

### 2. Imaging parameters

- `psf_size`
- `psf_sigma_xy`
- `psf_sigma_z`
- `photon_level`
- `read_noise_sigma`
- `apply_gaussian_noise`
- `gaussian_std`
- `background_strength`
- `band_strength`
- `gradient_strength`

---

### 3. Projection and mask parameters

- `projection`
- `slab_thickness`
- `mask_low_thr`
- `mask_high_thr`
- `mask_min_size`
- `seed_erode_iters`
- `mask_gamma`
- `max_hole_size`

---

### 4. Other parameters

- `shape`
- `model`
- `seed`

---

## Outputs

This script does not save files by default.

Its main output is a matplotlib figure with four subplots:

1. Raw mask
2. Visible mask
3. Clean image
4. Noisy image

It also displays the figure interactively using `plt.show()`.

---

## High-Level Logic

At a high level, the script works like this:

1. Generate one synthetic example using the full pipeline.
2. Convert the raw 3D mask into a 2D projected mask.
3. Create a 1×4 visualization figure.
4. Show:
   - raw projected mask,
   - final visible mask,
   - clean image,
   - noisy image.
5. Remove axes and display the result.

So this file is essentially a quick visual front-end for the generation pipeline.

---

## Main Function: `main(...)`

This is the core function of the script.

### What it does

- calls `generate_single_example(...)`,
- receives all intermediate outputs,
- projects the raw mask to 2D,
- builds a visualization figure,
- displays the four key outputs.

### Returned values from `generate_single_example(...)`

The function receives:

- `vol`
- `raw_mask`
- `psf`
- `clean`
- `noisy`
- `visible_mask`

However, in this file, only some of these are visualized directly.

### Pseudocode

```text
FUNCTION main(...)

    CALL generate_single_example(...) with all provided parameters

    RECEIVE:
        vol
        raw_mask
        psf
        clean
        noisy
        visible_mask

    CONVERT raw 3D mask into 2D mask using _mask_to_2d(...)
    STORE result as raw_mask_slice

    CREATE matplotlib figure with 1 row and 4 columns

    DISPLAY raw_mask_slice in grayscale
    SET title = "Raw mask"

    DISPLAY visible_mask in grayscale
    SET title = "Visible mask"

    DISPLAY clean image in grayscale
    SET title = "Clean image"

    DISPLAY noisy image in grayscale
    SET title = "Noisy image"

    FOR each subplot
        TURN axis off
    END FOR

    APPLY tight layout
    SHOW figure

END FUNCTION
```

---

## Visualization Layout

The visualization is arranged as:

```text
[ Raw mask ] [ Visible mask ] [ Clean image ] [ Noisy image ]
```

### Meaning of each panel

#### Raw mask
This is the projected geometry-based mask before visibility refinement.

#### Visible mask
This is the refined 2D mask built from the clean image and constrained by raw geometry.

#### Clean image
This is the rendered synthetic image after PSF-based imaging simulation but before final noise addition.

#### Noisy image
This is the final synthetic microscopy-like image after adding noise and background effects.

---

## Why this file is useful

This file is very useful during development because it lets you quickly inspect whether:

- the generated structure looks plausible,
- the projection mode behaves as expected,
- the visible mask follows the structure well,
- the noise settings are too strong or too weak,
- the clean/noisy relationship looks realistic.

It is basically a quick sanity-check tool for the synthetic pipeline.

---

## Command-Line Interface

At the bottom of the file, there is a command-line interface.

This means the script can be run directly from the terminal.

### CLI behavior

1. Create an argument parser.
2. Read selected user parameters.
3. Call `main(...)` with those parameter values.
4. Show the visualization.

---

## CLI Arguments

### Model and projection arguments

- `--model`
- `--projection`
- `--slab-thickness`

### Mask creation arguments

- `--mask-low-thr`
- `--mask-high-thr`
- `--mask-min-size`
- `--seed-erode-iters`
- `--mask-gamma`
- `--max-hole-size`

### Noise arguments

- `--apply-gaussian-noise`
- `--gaussian-std`
- `--background-strength`
- `--band-strength`
- `--gradient-strength`

### Randomness argument

- `--seed`

---

## CLI Pseudocode

```text
IF file is executed as main program THEN

    CREATE argument parser

    ADD argument --model
    ADD argument --projection
    ADD argument --slab-thickness

    ADD argument --mask-low-thr
    ADD argument --mask-high-thr
    ADD argument --mask-min-size
    ADD argument --seed-erode-iters
    ADD argument --mask-gamma
    ADD argument --max-hole-size

    ADD argument --apply-gaussian-noise
    ADD argument --gaussian-std
    ADD argument --background-strength
    ADD argument --band-strength
    ADD argument --gradient-strength

    ADD argument --seed

    PARSE command-line arguments

    CALL main(...) using parsed argument values

END IF
```

---

## Example Conceptual Workflow

The intended workflow of this file is:

```text
1. Choose model and rendering settings
2. Generate one sample
3. Compare:
   raw mask
   visible mask
   clean image
   noisy image
4. Decide whether parameters look good
5. If yes, move on to full dataset generation
6. If no, adjust parameters and test again
```

So this file is very helpful before running large dataset-generation jobs.

---

## Notes

- This file is a **demo / inspection tool**, not a dataset exporter.
- It visualizes one sample only.
- It relies on `dataset_generator.py` for the actual generation logic.
- It is a good first tool to use when testing new parameter combinations.
- It is especially useful when tuning:
  - projection settings,
  - mask thresholds,
  - visible-mask behavior,
  - background and noise strengths.

---

## Relationship to other project files

This file sits on top of the generation pipeline.

### Relationship summary

- `shapes.py`  
  creates synthetic 3D structures

- `psf.py`  
  defines the optical blur kernel

- `render.py`  
  converts the 3D structure into a 2D clean image

- `noise.py`  
  adds microscopy-like noise

- `dataset_generator.py`  
  combines all steps into one sample or full dataset generation

- `demo_pipeline.py`  
  shows a single generated example visually

So this file is the easiest visual entry point into the whole generation pipeline.

---

## Why this file is important in the project

This file is important because it gives an immediate visual explanation of how the synthetic-data pipeline behaves.

In the thesis context, it is useful because it helps demonstrate the intermediate stages of the pipeline:

- geometry-based mask,
- visibility-based mask,
- clean simulated image,
- final noisy image.

That makes it useful both for development and for explaining the pipeline to others.

In simple terms:

- `dataset_generator.py` builds the sample,
- `demo_pipeline.py` lets you see what was built.
