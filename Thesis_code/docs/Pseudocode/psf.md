# psf.py

## Purpose

This file contains the point spread function (PSF) utilities used in the synthetic microscopy pipeline.

Its purpose is to simulate optical blur by creating a 3D Gaussian PSF kernel and applying it to a 3D volume.

In microscopy, a PSF describes how a point source of light appears in the imaging system.  
Because real microscopes blur light instead of capturing perfectly sharp points, the synthetic pipeline needs a PSF step to make rendered structures look more realistic.

This file is therefore part of the optical imaging simulation stage of the project.

---

## Main Functions

This file contains three main functions:

1. `make_gaussian_psf(...)`
2. `apply_psf(...)`
3. `gaussian_psf_kernel(...)`

---

## High-Level Role in the Pipeline

At a high level, the file is used like this:

```text
1. Create a synthetic 3D intensity volume
2. Build a 3D Gaussian PSF kernel
3. Convolve the volume with the PSF
4. Obtain a blurred version of the volume
5. Continue with projection / rendering
```

So this file introduces optical blur into the synthetic image formation process.

---

## Inputs

### `make_gaussian_psf(...)`

#### Inputs
- `size`  
  Kernel size as `(Z, Y, X)`

- `sigma`  
  Standard deviations as `(sz, sy, sx)`

#### Output
- A normalized 3D Gaussian PSF kernel as a NumPy array

---

### `apply_psf(volume, psf)`

#### Inputs
- `volume`  
  3D image volume

- `psf`  
  3D PSF kernel

#### Output
- A blurred 3D volume with the same shape as the input volume

---

### `gaussian_psf_kernel(...)`

#### Inputs
- `size`
- `sigma_xy`
- `sigma_z`

#### Output
- A normalized 3D Gaussian kernel

This is mainly a convenience wrapper around `make_gaussian_psf(...)`.

---

## Function: `make_gaussian_psf(size=(17,17,9), sigma=(2.0,2.0,1.0))`

This function builds a 3D Gaussian PSF kernel.

### What it does

1. Read kernel dimensions.
2. Read Gaussian standard deviations.
3. Build coordinate axes centered around zero.
4. Create a 3D coordinate grid.
5. Evaluate the Gaussian function on that grid.
6. Normalize the kernel so its total sum is 1.
7. Return the result as `float32`.

### Why normalization matters

The PSF is normalized so that total energy stays constant after convolution.

That means:

- the image becomes blurred,
- but total intensity is preserved.

### Pseudocode

```text
FUNCTION make_gaussian_psf(size = (17, 17, 9), sigma = (2.0, 2.0, 1.0))

    EXTRACT z_size, y_size, x_size from size
    EXTRACT sz, sy, sx from sigma

    CREATE z coordinates centered around zero
    CREATE y coordinates centered around zero
    CREATE x coordinates centered around zero

    BUILD 3D coordinate grids Z, Y, X

    COMPUTE Gaussian kernel:
        exp(
            -(
                Z^2 / (2 * sz^2)
                + Y^2 / (2 * sy^2)
                + X^2 / (2 * sx^2)
            )
        )

    NORMALIZE kernel by dividing by its sum

    CONVERT kernel to float32

    RETURN kernel
END FUNCTION
```

---

## Function: `apply_psf(volume, psf)`

This function applies the PSF to a 3D volume.

### What it does

- performs a 3D convolution between the input volume and the PSF,
- uses FFT-based convolution for efficiency,
- returns a blurred volume with the same shape as the input.

### Why FFT convolution is used

FFT-based convolution is much faster than direct convolution for larger kernels and volumes.

That makes it a practical choice for repeated synthetic-data generation.

### Pseudocode

```text
FUNCTION apply_psf(volume, psf)

    COMPUTE FFT-based convolution of volume with psf
    USE mode = "same" so output shape matches input shape

    CONVERT result to float32

    RETURN blurred volume
END FUNCTION
```

---

## Function: `gaussian_psf_kernel(size=(17,17,9), sigma_xy=1.0, sigma_z=1.0)`

This function is a convenience wrapper.

It exists because some older parts of the code expect separate `sigma_xy` and `sigma_z` arguments instead of one sigma tuple.

### What it does

- converts `(sigma_xy, sigma_z)` style arguments into the format expected by `make_gaussian_psf(...)`,
- calls `make_gaussian_psf(...)`,
- returns the resulting kernel.

### Important mapping

The wrapper maps:

```text
sigma = (sigma_z, sigma_xy, sigma_xy)
```

because:

- `Z` uses `sigma_z`
- `Y` and `X` both use `sigma_xy`

### Pseudocode

```text
FUNCTION gaussian_psf_kernel(size = (17, 17, 9), sigma_xy = 1.0, sigma_z = 1.0)

    CONSTRUCT sigma tuple as:
        (sigma_z, sigma_xy, sigma_xy)

    CALL make_gaussian_psf(size = size, sigma = constructed tuple)

    RETURN resulting kernel
END FUNCTION
```

---

## Conceptual Meaning of the PSF

In a perfect imaging system, a single bright point would appear as a single perfect point.

In a real microscope, that point is blurred.

The PSF describes exactly that blur pattern.

In this file, the PSF is approximated as a 3D Gaussian.

That means the blur is strongest at the center and smoothly falls off in all directions.

Because `sigma_z` and `sigma_xy` can differ, the blur can be anisotropic:

- axial blur in `Z` may differ from lateral blur in `X/Y`.

This is important because microscopy systems often have different resolution in axial and lateral directions.

---

## Relationship to the Pipeline

This file is used after a 3D synthetic structure volume has been created.

Typical pipeline step:

```text
3D structure volume
→ PSF blur
→ blurred 3D volume
→ 2D projection / rendering
```

So this file is one of the key parts of the forward imaging model.

---

## Relationship to Other Project Files

### Relationship summary

- `shapes.py`  
  creates the 3D synthetic structures

- `psf.py`  
  creates and applies the optical blur

- `render.py`  
  uses the blurred volume for 2D image rendering

- `dataset_generator.py`  
  calls `gaussian_psf_kernel(...)` when generating samples

So `psf.py` sits between structure generation and final rendering.

---

## Notes

- The PSF kernel is always normalized.
- `apply_psf(...)` uses FFT convolution from `scipy.signal`.
- The output volume has the same shape as the input volume.
- The convenience wrapper exists for API compatibility with older pipeline code.
- The kernel dimensions are given in `(Z, Y, X)` order.
- The Gaussian sigmas are also interpreted in `(Z, Y, X)` order when passed as a tuple.

---

## Why this file is important in the thesis project

This file is important because a major part of the thesis is not only to generate synthetic structures, but to generate realistic microscopy images.

A clean geometric volume by itself is too idealized.

The PSF step introduces the optical limitations of the imaging system, making the result more physically plausible.

Without this step:

- structures would look too sharp,
- images would not resemble real microscopy well enough.

In simple terms:

- `shapes.py` creates the object,
- `psf.py` simulates optical blur,
- `render.py` turns that blurred object into an image.

So `psf.py` is one of the files that makes the synthetic data physically informed rather than purely geometric.
