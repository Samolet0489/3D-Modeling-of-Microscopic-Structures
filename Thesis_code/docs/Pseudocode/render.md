# render.py

## Purpose

This file contains the rendering utilities used to convert a synthetic 3D volume into a 2D microscopy-style image.

Its purpose is to:

- blur a 3D volume using a PSF,
- project the blurred 3D volume into 2D,
- support different projection modes,
- provide a simple middle-slice helper for visualization.

This file is part of the image-formation stage of the project.

It sits between:

- the generation of a 3D synthetic structure volume, and
- the addition of final image noise.

So this file is responsible for turning a 3D object into a 2D image-like representation.

---

## Main Functions

This file contains three functions:

1. `project_volume(...)`
2. `get_middle_slice(...)`
3. `render_image(...)`

---

## High-Level Role in the Pipeline

At a high level, this file works like this:

```text
1. Start with a 3D volume
2. Blur the volume using a PSF
3. Convert the blurred 3D volume into a 2D image
4. Return the 2D result
```

This means `render.py` is the step where a 3D synthetic structure becomes a 2D microscopy observation.

---

## Inputs

### `project_volume(volume, mode="max", slab_thickness=5)`

#### Inputs
- `volume`  
  3D array with shape `(Z, Y, X)`

- `mode`  
  projection mode:
  - `"max"`
  - `"sum"`
  - `"focus_max"`

- `slab_thickness`  
  thickness of the central slab used in `"focus_max"`

#### Output
- a 2D projection of shape `(Y, X)`

---

### `get_middle_slice(volume)`

#### Input
- `volume`  
  3D array with shape `(Z, Y, X)`

#### Output
- the middle z-slice as a 2D array of shape `(Y, X)`

---

### `render_image(volume, psf, projection="center_slice", slab_thickness=5)`

#### Inputs
- `volume`  
  3D synthetic intensity volume

- `psf`  
  3D PSF kernel

- `projection`  
  output method:
  - `"center_slice"`
  - or one of the projection modes handled by `project_volume(...)`

- `slab_thickness`  
  slab thickness used when projection is `"focus_max"`

#### Output
- a 2D rendered image

---

## Function: `project_volume(volume, mode="max", slab_thickness=5)`

This function converts a 3D volume into a 2D image by projection.

### Supported modes

#### `"max"`
Take the maximum intensity along the z-axis.

#### `"sum"`
Sum intensities along the z-axis.

#### `"focus_max"`
Take a maximum-intensity projection only over a centered z-slab instead of the whole volume.

### What it does

- checks the projection mode,
- chooses the corresponding reduction over z,
- returns the 2D result.

### Pseudocode

```text
FUNCTION project_volume(volume, mode = "max", slab_thickness = 5)

    IF mode is "max" THEN
        RETURN maximum of volume along z axis
    END IF

    IF mode is "sum" THEN
        RETURN sum of volume along z axis
    END IF

    IF mode is "focus_max" THEN
        ENSURE slab_thickness is at least 1

        FIND middle z index
        COMPUTE half slab thickness
        COMPUTE start index z0
        COMPUTE end index z1

        EXTRACT centered z slab
        RETURN maximum of slab along z axis
    END IF

    RAISE error for unknown mode
END FUNCTION
```

---

## Function: `get_middle_slice(volume)`

This function returns the middle z-slice of a 3D volume.

### What it does

- computes the middle z-index,
- extracts that slice,
- returns it.

This helper is mostly useful for visualization and debugging.

### Pseudocode

```text
FUNCTION get_middle_slice(volume)

    COMPUTE middle z index

    RETURN volume at that z index
END FUNCTION
```

---

## Function: `render_image(volume, psf, projection="center_slice", slab_thickness=5)`

This is the main rendering function in the file.

### What it does

1. Blur the input 3D volume using the PSF.
2. Convert the blurred volume into a 2D image.
3. Return the 2D result.

### Projection behavior

If `projection == "center_slice"`:

- return the middle slice of the blurred volume.

Otherwise:

- call `project_volume(...)` with the chosen projection mode.

### Pseudocode

```text
FUNCTION render_image(volume, psf, projection = "center_slice", slab_thickness = 5)

    APPLY PSF blur to volume using apply_psf(...)
    STORE result as blurred

    IF projection is "center_slice" THEN
        FIND middle z index
        RETURN middle slice of blurred volume
    END IF

    CALL project_volume(
        blurred,
        mode = projection,
        slab_thickness = slab_thickness
    )

    RETURN projected 2D image
END FUNCTION
```

---

## Conceptual Meaning of the Rendering Step

This file models the step from:

```text
synthetic 3D object
→ blurred optical signal
→ 2D image observation
```

That is important because the final segmentation model is trained on 2D images, but the synthetic objects are modeled in 3D.

So this file is one of the key pieces that bridges:

- 3D structure generation
- 2D training data creation

---

## Projection Modes Explained Simply

### `center_slice`
Take only the middle layer of the blurred volume.

This can simulate observing a single focal plane.

---

### `max`
Take the brightest value across the entire z-axis for each pixel.

This produces a maximum-intensity projection of the full volume.

---

### `sum`
Add intensities across the full z-axis.

This produces a summed projection.

---

### `focus_max`
Take only a limited slab around the middle of the volume, then do max projection there.

This is a compromise between:

- full-volume projection
- single-slice observation

It can help simulate a more focused image while still capturing nearby structure.

---

## Relationship to Other Project Files

This file depends on:

- `apply_psf(...)` from `psf.py`

It is used by:

- `dataset_generator.py`
- `demo_pipeline.py`

### Relationship summary

- `shapes.py`  
  creates the 3D structure

- `psf.py`  
  blurs the structure

- `render.py`  
  turns the blurred 3D volume into a 2D image

- `noise.py`  
  adds final image noise

So `render.py` is the stage where the blurred 3D scene is turned into the 2D signal used later in the dataset.

---

## Notes

- Input volumes are expected in `(Z, Y, X)` order.
- The output image is always 2D with shape `(Y, X)`.
- `render_image(...)` first applies PSF blur, then projection.
- `project_volume(...)` does not apply blur by itself.
- `get_middle_slice(...)` is a small helper mainly for debugging or visualization.

---

## Why this file is important in the thesis project

This file is important because the project starts from 3D synthetic biological structures, but the initial training task is 2D segmentation.

That means there must be a clear and controlled way to turn 3D synthetic structures into 2D image data.

This file performs exactly that step.

Without this file:

- you could generate 3D structures,
- but you would not yet have realistic 2D training images.

In simple terms:

- `shapes.py` creates the object,
- `psf.py` blurs it,
- `render.py` turns it into a 2D image,
- `noise.py` makes the image look realistic.

So `render.py` is one of the central image-formation files in the project.
