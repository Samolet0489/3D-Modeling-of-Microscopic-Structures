# noise.py

## Purpose

This file contains the noise-generation utilities used to make synthetic microscopy images look more realistic.

Its purpose is to simulate the kinds of intensity variation and corruption that appear in microscopy images, such as:

- Poisson shot noise,
- Gaussian read noise,
- structured background,
- weak banding,
- global illumination gradients.

This file is part of the imaging realism side of the project.  
It does not create biological structures or masks.  
Instead, it takes an already generated image and makes it look more like a real microscope acquisition.

---

## Main Functions

This file contains the following important functions:

1. `add_poisson_noise(...)`
2. `add_gaussian_noise(...)`
3. `_make_structured_background(...)`
4. `_supervisor_style_noise(...)`
5. `add_microscopy_noise(...)`
6. `add_noise(...)`

Some of these are marked as legacy helpers, while others are the main functions currently used by the pipeline.

---

## High-Level Role in the Pipeline

At a high level, this file is used after a clean synthetic image has already been rendered.

Typical workflow:

```text
1. Generate synthetic structure
2. Render clean image
3. Add structured background
4. Add Poisson noise
5. Optionally add Gaussian noise
6. Return final noisy microscopy-like image
```

So this file is responsible for the realism of the final image appearance.

---

## Inputs

The main inputs across the functions are:

- `image`  
  a 2D float image array

- `seed`  
  random seed for reproducibility

- `peak_photons` / `photon_level`  
  controls Poisson noise scaling

- `read_sigma` / `read_noise_sigma` / `std`  
  controls Gaussian noise strength

- `background_strength`  
  controls cloudy low-frequency background

- `band_strength`  
  controls weak column-wise banding

- `gradient_strength`  
  controls a weak illumination gradient

- `return_uint8`  
  decides whether to return the final image as uint8 or float

---

## Outputs

Depending on the function, the output is:

- a noisy float image in `[0, 1]`, or
- a noisy uint8 image in `[0, 255]`.

The main default behavior is to return float images in `[0, 1]`.

---

## Function: `add_poisson_noise(image, peak_photons=5000, seed=None)`

This is a legacy helper function.

### What it does

- makes the image non-negative,
- normalizes it to `[0, 1]`,
- scales intensities into photon counts,
- samples Poisson noise,
- rescales back to `[0, 1]`.

This simulates shot noise caused by the discrete nature of photons.

### Pseudocode

```text
FUNCTION add_poisson_noise(image, peak_photons = 5000, seed = None)

    CREATE random number generator

    CONVERT image to float32
    SHIFT image so minimum becomes zero

    IF image maximum > 0 THEN
        NORMALIZE image to [0, 1]
    END IF

    SCALE image by peak_photons

    SAMPLE Poisson noise from scaled image

    DIVIDE noisy photon counts by peak_photons

    RETURN noisy image in [0, 1]
END FUNCTION
```

---

## Function: `add_gaussian_noise(image, sigma=0.03, seed=None)`

This is another legacy helper.

### What it does

- samples zero-mean Gaussian noise,
- adds it to the image,
- clips the result to `[0, 1]`.

This simulates read noise or general additive sensor noise.

### Pseudocode

```text
FUNCTION add_gaussian_noise(image, sigma = 0.03, seed = None)

    CREATE random number generator

    CONVERT image to float32

    SAMPLE Gaussian noise with mean 0 and std = sigma

    ADD noise to image

    CLIP result to [0, 1]

    RETURN noisy image
END FUNCTION
```

---

## Function: `_make_structured_background(...)`

This helper creates a weak microscopy-like background pattern.

It combines three components:

1. low-frequency cloudy background
2. soft column-wise banding
3. weak directional gradient

This is important because real microscope backgrounds are often not perfectly flat.

### Inputs

- `shape`
- `rng`
- `background_strength`
- `band_strength`
- `gradient_strength`

### Pseudocode

```text
FUNCTION _make_structured_background(
    shape,
    rng,
    background_strength = 0.08,
    band_strength = 0.025,
    gradient_strength = 0.03
)

    EXTRACT height and width from shape

    CREATE random 2D field
    APPLY strong Gaussian smoothing
    NORMALIZE field to zero mean and unit std

    CREATE random 1D column signal
    APPLY Gaussian smoothing to column signal
    NORMALIZE column signal
    TILE it across rows to make band image

    CREATE normalized x and y coordinate grids

    SAMPLE random gradient direction angle
    BUILD directional gradient
    NORMALIZE gradient

    COMBINE:
        background_strength * cloudy field
        + band_strength * band image
        + gradient_strength * gradient

    RETURN background image as float32
END FUNCTION
```

---

## Function: `_supervisor_style_noise(...)`

This is an important helper that applies the main “supervisor-style” noise model.

### What it does

1. shift image so it is non-negative
2. choose random offset and max signal values
3. normalize image into that signal range
4. apply Poisson noise
5. optionally apply Gaussian noise
6. clip to valid intensity range
7. return either uint8 or float

### Important idea

This function does not just apply noise directly to a normalized image.  
It first maps the image into a realistic signal range using:

- a random offset,
- a random maximum signal level.

That helps simulate more realistic acquisition intensity behavior.

### Pseudocode

```text
FUNCTION _supervisor_style_noise(
    image,
    rng,
    apply_gaussian = False,
    std = 15.0,
    offset_range = (40, 70),
    max_signal_range = (200, 240),
    return_uint8 = False
)

    CONVERT image to float32

    SHIFT image so minimum becomes zero
    COMPUTE image peak

    SAMPLE random offset from offset_range
    SAMPLE random max signal from max_signal_range

    IF peak <= 0 THEN
        FILL normalized image with offset value
    ELSE
        MAP image into range [offset, max_signal]
    END IF

    SAMPLE Poisson noise from normalized image

    IF Gaussian noise is enabled THEN
        SAMPLE Gaussian noise around the Poisson result
    END IF

    CLIP intensities to [0, 255]

    IF return_uint8 is True THEN
        RETURN uint8 image
    ELSE
        RETURN float image normalized to [0, 1]
    END IF
END FUNCTION
```

---

## Function: `add_microscopy_noise(...)`

This is a convenience wrapper for applying the supervisor-style noise model.

### What it does

- creates a random number generator,
- optionally converts `read_sigma` into pixel-scale Gaussian std,
- calls `_supervisor_style_noise(...)`.

### Important note

Gaussian noise is off by default.

### Pseudocode

```text
FUNCTION add_microscopy_noise(
    image,
    peak_photons = 5000,
    read_sigma = 0.03,
    seed = None,
    apply_gaussian = False,
    std = 15.0,
    return_uint8 = False
)

    CREATE random number generator

    IF Gaussian is enabled AND std still has default value AND read_sigma exists THEN
        IF read_sigma <= 1.0 THEN
            CONVERT read_sigma from normalized scale to uint8 scale
        ELSE
            USE read_sigma directly as std
        END IF
    END IF

    CALL _supervisor_style_noise(...)
    RETURN result
END FUNCTION
```

---

## Function: `add_noise(...)`

This is the most important high-level noise helper in the file.

It matches the API used by the demo pipeline and the dataset generation code.

### What it does

1. create a random number generator,
2. optionally interpret `photon_level` as a fixed signal level,
3. optionally interpret `read_noise_sigma` as Gaussian std,
4. optionally create structured background,
5. add that background to the clean image,
6. call `_supervisor_style_noise(...)`,
7. return the final noisy result.

### Key point

This function combines:

- structured background realism
- supervisor-style Poisson noise
- optional Gaussian noise

So this is effectively the main “realistic noise” entry point used by the project.

### Pseudocode

```text
FUNCTION add_noise(
    image,
    photon_level = 2000.0,
    read_noise_sigma = 0.03,
    seed = None,
    apply_gaussian = False,
    std = 15.0,
    background_strength = 0.08,
    band_strength = 0.025,
    gradient_strength = 0.03,
    return_uint8 = False
)

    CREATE random number generator

    SET default offset range = (40, 70)
    SET default max signal range = (200, 240)

    IF photon_level is between 1 and 255 THEN
        USE photon_level as fixed max signal
    END IF

    IF Gaussian is enabled AND std still default AND read_noise_sigma exists THEN
        IF read_noise_sigma <= 1.0 THEN
            CONVERT read_noise_sigma to uint8-scale std
        ELSE
            USE read_noise_sigma directly
        END IF
    END IF

    COPY input image

    IF any structured background strength is greater than zero THEN
        BUILD background using _make_structured_background(...)
        ADD background to image
        CLIP so image does not go below zero
    END IF

    CALL _supervisor_style_noise(...) using:
        modified image
        random generator
        Gaussian settings
        chosen intensity ranges
        return type setting

    RETURN final noisy image
END FUNCTION
```

---

## Relationship Between the Functions

The functions can be understood in layers.

### Legacy simple helpers
- `add_poisson_noise(...)`
- `add_gaussian_noise(...)`

These are simpler standalone noise functions.

### Background helper
- `_make_structured_background(...)`

This creates realistic low-frequency background artifacts.

### Core noise model
- `_supervisor_style_noise(...)`

This is the main low-level microscopy noise engine.

### High-level wrappers
- `add_microscopy_noise(...)`
- `add_noise(...)`

These expose the noise model in easier-to-use forms.

---

## Why structured background matters

A perfectly clean background often looks unrealistic in microscopy.

This file improves realism by adding:

- cloudy intensity drift,
- weak banding,
- smooth directional gradients.

That makes the final synthetic images look more like real acquisitions rather than idealized renderings.

Conceptually:

```text
clean image
+ structured background
+ Poisson shot noise
+ optional Gaussian read noise
= final microscopy-like image
```

---

## Notes

- The main default output is float `[0, 1]`.
- The code also supports returning uint8 when needed.
- `photon_level` has a compatibility behavior:
  - if passed in `[1, 255]`, it is interpreted as a fixed max signal level.
- Gaussian noise is optional.
- The supervisor-style noise is the preferred modern approach in this file.
- The structured background is applied before Poisson and Gaussian noise.
- The background is clipped so the image does not go negative before noise is applied.

---

## Role in the Thesis Project

This file is important because image realism is one of the core goals of the thesis.

The project is not only about generating synthetic structures, but about making the final images realistic enough that they are useful for AI training.

That means the final image must resemble real microscopy, including:

- sensor noise,
- photon statistics,
- non-flat background,
- weak acquisition artifacts.

This file is where that realism is introduced.

---

## Relationship to Other Project Files

This file works together with the rest of the synthetic image pipeline.

### Relationship summary

- `shapes.py`  
  creates the synthetic 3D structures

- `render.py`  
  produces the clean projected image

- `noise.py`  
  adds microscopy-like realism

- `dataset_generator.py`  
  uses `add_noise(...)` to generate final dataset images

- `demo_pipeline.py`  
  visualizes the clean and noisy results side by side

So this file is the step that turns a clean synthetic rendering into a more realistic microscope-style image.

---

## Why this file is important in simple words

In simple terms:

- without this file, the images would look too clean,
- with this file, the images look more like real microscope images.

That makes this file very important for bridging the gap between synthetic data and real-world training data.
