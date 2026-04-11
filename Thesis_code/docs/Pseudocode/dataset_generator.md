# dataset_generator.py

## Purpose

This file contains the dataset generation utilities for Phase 1 (2.5D) of the project.

Its main job is to generate synthetic microscopy image/mask pairs and save them in a dataset folder that can later be used for training 2D segmentation models.

The module supports:

- generating one synthetic example,
- generating and saving a full dataset,
- creating masks from projected 3D data,
- applying quality control (QC) checks,
- saving metadata such as run settings and per-sample parameters,
- running the whole pipeline from the command line.

This file is one of the core parts of the synthetic-data pipeline because it connects structure generation, rendering, noise simulation, mask extraction, and dataset export.

---

## Main Dependencies

This file depends on other project modules:

- `shapes.py`  
  for generating synthetic 3D mitochondrial volumes

- `psf.py`  
  for building the Gaussian PSF kernel

- `render.py`  
  for turning 3D intensity volumes into 2D projected images

- `noise.py`  
  for adding microscopy-style noise to the clean rendered images

It also uses:

- `numpy`
- `imageio.imwrite`
- `scipy.ndimage.binary_propagation`
- `scipy.ndimage.label`
- `json`
- `argparse`
- `os`

---

## Inputs

### Main functional inputs

The module has many parameters, but they can be grouped into the following categories.

### 1. Output and dataset settings

- `out_dir`  
  folder where the dataset will be saved

- `n_samples`  
  number of image/mask pairs to generate

- `seed`  
  random seed for reproducibility

---

### 2. Volume / structure settings

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

### 3. Imaging settings

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

### 4. Projection and mask settings

- `projection`
- `slab_thickness`
- `mask_low_thr`
- `mask_high_thr`
- `mask_min_size`
- `seed_erode_iters`
- `mask_gamma`
- `max_hole_size`

---

### 5. Quality control settings

- `min_fg_fraction`
- `max_fg_fraction`
- `max_border_fraction`
- `border_width`
- `max_attempts_per_sample`

---

### 6. Behavior setting

- `randomize_per_sample`  
  decides whether volume and imaging parameters stay fixed or are resampled for each sample

---

## Outputs

The module produces the following outputs.

### Per-sample outputs

For each generated sample:

- one PNG image in `images/`
- one PNG mask in `masks/`

### Run-level outputs

For the whole dataset:

- `run_config.json`  
  stores the full run configuration

- `sample_manifest.jsonl`  
  stores one metadata record per saved sample

### Function return outputs

`generate_single_example(...)` returns:

- `volume`
- `mask`
- `psf`
- `clean`
- `noisy`
- `mask2d`

`generate_and_save_dataset(...)` mainly saves files to disk and prints progress information.

---

## High-Level Logic

At a high level, the file works like this:

1. Build or sample structure parameters.
2. Build or sample imaging parameters.
3. Generate a synthetic 3D structure volume and 3D raw mask.
4. Build a PSF kernel.
5. Render the 3D volume into a 2D clean microscopy-like image.
6. Convert the 3D mask into a 2D projected mask.
7. Refine the 2D mask using visibility-based thresholding.
8. Add realistic imaging noise to the clean image.
9. Evaluate mask quality with QC rules.
10. If the mask passes QC, save image, mask, and metadata.
11. Repeat until the requested number of samples is produced.

---

## Important Helper Functions

This file has several helper functions. Each one handles one small part of the pipeline.

---

### `_mask_to_2d(mask, projection, slab_thickness=5)`

Converts a 3D mask into a 2D mask using the chosen projection mode.

Supported modes:

- `center_slice`
- `max`
- `sum`
- `focus_max`

#### Pseudocode

```text
FUNCTION _mask_to_2d(mask, projection, slab_thickness = 5)

    IF projection is "center_slice" THEN
        TAKE the middle z slice
        RETURN that 2D slice
    END IF

    IF projection is "max" THEN
        TAKE maximum along z axis
        RETURN 2D result
    END IF

    IF projection is "sum" THEN
        SUM along z axis
        RETURN whether sum is greater than zero
    END IF

    IF projection is "focus_max" THEN
        COMPUTE a centered z slab around the middle slice
        TAKE maximum over that slab
        RETURN 2D result
    END IF

    RAISE error for unknown projection
END FUNCTION
```

---

### `_write_json(path, data)`

Writes a Python object as formatted JSON.

```text
FUNCTION _write_json(path, data)
    OPEN file for writing
    SERIALIZE data as JSON with indentation
    SAVE file
END FUNCTION
```

---

### `_append_jsonl(path, record)`

Appends one JSON record to a JSONL file.

```text
FUNCTION _append_jsonl(path, record)
    OPEN file in append mode
    SERIALIZE record as one JSON line
    APPEND newline
END FUNCTION
```

---

### `_build_fixed_volume_params(...)`

Builds a dictionary of fixed structure parameters based on the chosen model.

- For Model A, it stores tube-based parameters.
- For Model B, it stores random-walk / polyline parameters.

```text
FUNCTION _build_fixed_volume_params(...)

    IF model is "A" THEN
        RETURN dictionary with n_tubes and tube_radius
    ELSE
        RETURN dictionary with Model B parameters
    END IF
END FUNCTION
```

---

### `_sample_randomized_volume_params(model, rng)`

Randomly samples structure-generation parameters.

```text
FUNCTION _sample_randomized_volume_params(model, rng)

    IF model is "A" THEN
        RANDOMLY sample n_tubes
        RANDOMLY sample tube_radius
        RETURN dictionary
    ELSE
        RANDOMLY sample Model B parameters
        RETURN dictionary
    END IF
END FUNCTION
```

---

### `_build_fixed_imaging_params(...)`

Builds a dictionary of fixed imaging parameters.

```text
FUNCTION _build_fixed_imaging_params(...)
    RETURN dictionary with PSF, photon, noise, and background parameters
END FUNCTION
```

---

### `_sample_randomized_imaging_params(rng)`

Randomly samples imaging parameters.

```text
FUNCTION _sample_randomized_imaging_params(rng)
    RANDOMLY sample PSF parameters
    RANDOMLY sample photon and noise parameters
    RANDOMLY sample background parameters
    RETURN dictionary
END FUNCTION
```

---

### `_make_border_mask(shape_2d, border_width=2)`

Creates a boolean mask marking the outer image border.

```text
FUNCTION _make_border_mask(shape_2d, border_width = 2)

    CREATE 2D array of False
    MARK top border as True
    MARK bottom border as True
    MARK left border as True
    MARK right border as True

    RETURN border mask
END FUNCTION
```

---

### `_remove_small_components(mask2d, min_size=10)`

Removes tiny connected components from a binary mask.

```text
FUNCTION _remove_small_components(mask2d, min_size = 10)

    LABEL connected components

    IF no components exist THEN
        RETURN mask as boolean
    END IF

    COUNT pixels in each component
    KEEP only components with size >= min_size
    REMOVE background label

    RETURN cleaned mask
END FUNCTION
```

---

### `_fill_small_holes(mask2d, max_hole_size=0)`

Fills only very small internal holes in the mask.

This is intentionally conservative.

```text
FUNCTION _fill_small_holes(mask2d, max_hole_size = 0)

    IF max_hole_size <= 0 THEN
        RETURN mask unchanged
    END IF

    INVERT mask
    LABEL connected empty regions

    IDENTIFY which empty regions touch the image border

    FOR each empty region that does not touch the border
        IF region size <= max_hole_size THEN
            FILL hole in original mask
        END IF
    END FOR

    RETURN filled mask
END FUNCTION
```

---

### `_visible_mask_from_clean(...)`

Builds a 2D visible mask from the clean rendered image while constraining it by the raw projected geometry.

This is a very important helper because it transforms the ideal geometric mask into a more realistic visible mask.

Main ideas:

- normalize clean image
- suppress faint halos with gamma correction
- create strong seed regions
- create weak support regions
- propagate strong seeds within weak support
- fill only tiny holes
- remove tiny components

```text
FUNCTION _visible_mask_from_clean(clean, raw_mask2d, ...)

    CONVERT clean image to float
    SHIFT image so minimum becomes zero
    NORMALIZE image to [0, 1] if possible

    APPLY gamma correction

    CONVERT raw projected mask to boolean

    IF seed erosion is enabled THEN
        ITERATIVELY erode raw mask to create strong support
    ELSE
        strong_support = raw mask
    END IF

    strong = pixels above high threshold AND inside strong_support
    weak   = pixels above low threshold AND inside raw mask

    IF strong is empty THEN
        mask = weak
    ELSE
        GROW strong seeds inside weak support using binary propagation
    END IF

    FILL tiny holes
    REMOVE tiny components

    RETURN final boolean mask
END FUNCTION
```

---

### `_evaluate_mask_qc(...)`

Checks whether a 2D mask is acceptable for saving.

QC rules:

- reject masks that are too empty
- reject masks that are too full
- reject masks with too much border-touching foreground

```text
FUNCTION _evaluate_mask_qc(mask2d, ...)

    CONVERT mask to boolean

    COMPUTE:
        foreground pixel count
        foreground fraction

    IF mask is empty THEN
        border fraction = 1.0
    ELSE
        BUILD border mask
        COUNT foreground pixels on border
        COMPUTE border fraction
    END IF

    CREATE empty reasons list

    IF foreground fraction is below minimum THEN
        ADD "too_empty"
    END IF

    IF foreground fraction is above maximum THEN
        ADD "too_full"
    END IF

    IF border fraction is above threshold THEN
        ADD "too_border_heavy"
    END IF

    BUILD metrics dictionary:
        fg_pixels
        fg_fraction
        border_pixels
        border_fraction
        accepted
        reasons

    RETURN metrics
END FUNCTION
```

---

## Main Function: `generate_single_example(...)`

This function creates one synthetic training example.

### What it does

1. Generate one 3D synthetic structure volume and its raw 3D mask.
2. Build a Gaussian PSF.
3. Render the clean 2D image.
4. Normalize the clean image.
5. Project the raw 3D mask to 2D.
6. Create the visible 2D mask from the clean image.
7. Add noise to produce the final noisy image.
8. Return all intermediate and final outputs.

### Pseudocode

```text
FUNCTION generate_single_example(...)

    IF model is "A" THEN
        CALL generate_mito_volume(...)
        GET volume and mask
    ELSE IF model is "B" THEN
        CALL generate_mito_volume_polyline(...)
        GET volume and mask
    ELSE
        RAISE error
    END IF

    BUILD Gaussian PSF kernel

    RENDER clean 2D image from 3D volume using PSF and chosen projection

    NORMALIZE clean image to [0, 1]

    PROJECT raw 3D mask into 2D using _mask_to_2d(...)

    BUILD visible 2D mask from clean image using _visible_mask_from_clean(...)

    ADD imaging noise to clean image using add_noise(...)

    RETURN:
        volume,
        mask,
        psf,
        clean,
        noisy,
        mask2d
END FUNCTION
```

---

## Main Function: `generate_and_save_dataset(...)`

This is the main dataset production function.

### What it does

It generates `n_samples` acceptable samples and saves them as PNG files.

For each sample:

- it may regenerate multiple times,
- each attempt is checked with QC,
- only accepted masks/images are saved.

### Saved folder structure

```text
out_dir/
├─ images/
├─ masks/
├─ run_config.json
└─ sample_manifest.jsonl
```

### Pseudocode

```text
FUNCTION generate_and_save_dataset(out_dir, n_samples = 100, ...)

    CREATE images_dir = out_dir / "images"
    CREATE masks_dir  = out_dir / "masks"
    ENSURE both folders exist

    IF max_fg_fraction <= 0 THEN
        DISABLE upper foreground fraction limit
    END IF

    CREATE random number generator with seed

    BUILD run_config dictionary containing:
        output settings
        shape and PSF settings
        projection settings
        mask settings
        model choice
        randomization mode
        fixed volume parameters
        fixed imaging parameters
        QC settings

    SAVE run_config.json

    DEFINE path for sample_manifest.jsonl
    DELETE old manifest if it already exists

    PRINT run configuration
    PRINT generation start message

    STORE fixed parameter dictionaries

    FOR each sample index from 0 to n_samples - 1

        SET accepted = False
        INITIALIZE variables for last attempt information

        FOR each attempt from 1 to max_attempts_per_sample

            SAMPLE a per-attempt seed

            IF randomize_per_sample is enabled THEN
                SAMPLE new volume parameters
                SAMPLE new imaging parameters
            ELSE
                USE fixed volume parameters
                USE fixed imaging parameters
            END IF

            CALL generate_single_example(...)
            GET:
                volume,
                mask,
                psf,
                clean,
                noisy,
                mask2d

            EVALUATE mask quality using _evaluate_mask_qc(...)

            PRINT attempt summary:
                sample index
                attempt number
                seed
                foreground fraction
                border fraction
                accepted flag

            STORE last attempt information

            IF QC says accepted THEN
                accepted = True
                BREAK out of attempt loop
            END IF

        END FOR

        IF sample was never accepted THEN
            RAISE runtime error with last QC info
        END IF

        BUILD image file path
        BUILD mask file path

        SAVE noisy image as uint8 PNG
        SAVE mask as binary uint8 PNG

        BUILD metadata record with:
            index
            seed
            model
            projection
            mask parameters
            file paths
            used volume parameters
            used imaging parameters
            QC result

        APPEND metadata record to sample_manifest.jsonl

    END FOR

    PRINT final save messages:
        dataset path
        config path
        manifest path
END FUNCTION
```

---

## Command-Line Interface

When the file is run directly, it acts as a CLI tool for dataset generation.

### Main CLI behavior

1. Build an argument parser.
2. Read output path and all generation parameters.
3. Parse arguments.
4. Call `generate_and_save_dataset(...)` with those values.

### Main CLI Pseudocode

```text
IF file is executed as main program THEN

    CREATE argument parser

    ADD positional argument:
        out_dir

    ADD general dataset arguments:
        --n
        --model
        --projection
        --slab-thickness
        --mask-low-thr
        --mask-high-thr
        --mask-min-size
        --seed-erode-iters
        --mask-gamma
        --max-hole-size
        --seed

    ADD imaging arguments:
        --psf-sigma-xy
        --psf-sigma-z
        --photon-level
        --read-noise-sigma
        --apply-gaussian-noise
        --gaussian-std
        --background-strength
        --band-strength
        --gradient-strength

    ADD Model A arguments:
        --n-tubes
        --tube-radius

    ADD Model B arguments:
        --n-seeds
        --step-length
        --n-steps
        --branch-prob
        --branch-steps
        --radius-min
        --radius-max
        --persistence
        --z-scale
        --seed-spread

    ADD behavior argument:
        --randomize-per-sample

    ADD QC arguments:
        --min-fg-fraction
        --max-fg-fraction
        --max-border-fraction
        --border-width
        --max-attempts-per-sample

    PARSE command-line arguments

    CALL generate_and_save_dataset(...) using parsed arguments

END IF
```

---

## Notes

- This module supports two structure models:
  - **Model A** = simpler baseline tube model
  - **Model B** = more advanced random-walk / polyline model

- The file separates:
  - structure generation,
  - imaging simulation,
  - mask generation,
  - QC filtering,
  - dataset saving.

- The QC loop is important because not every generated mask is suitable for training.

- `sample_manifest.jsonl` is very useful for reproducibility because it records the exact parameters used for each saved sample.

- `run_config.json` stores the overall run settings for the whole dataset.

- `randomize_per_sample=True` is especially useful when building a more diverse dataset.

- The visible mask is not just a direct projection of geometry. It is constrained by the projected geometry but also shaped by image visibility rules.

---

## Why this file is important in the project

This is one of the most important files in the whole synthetic-data pipeline.

It is the file that transforms the project from a simple image-generation experiment into a reusable dataset-generation system.

Its role in the thesis project is:

- to create training-ready synthetic datasets,
- to connect 3D structure modeling with 2D segmentation data,
- to enforce quality control before saving samples,
- to record generation parameters for reproducibility,
- to support experiments comparing different models and settings.

In simple terms:

- `shapes.py` creates structure,
- `psf.py` and `render.py` simulate imaging,
- `noise.py` makes images more realistic,
- `dataset_generator.py` turns all of that into an actual dataset you can train on.
