# viz_3d.py

## Purpose

This file contains a visualization script for inspecting the generated 3D synthetic structures.

Its purpose is to help the user visually check the 3D mask and volume produced by the shape generators.

It creates several views of the generated structure:

- orthogonal 2D projections,
- a montage of z-slices,
- a 3D scatter plot of mask voxels,
- a small info panel with generation details.

This file is mainly a debugging and inspection tool.  
It is useful for confirming that the generated 3D structures look reasonable before continuing with rendering, dataset generation, or model training.

---

## Main Responsibilities

This file is responsible for:

1. generating a 3D synthetic structure using Model A or Model B,
2. extracting useful visualization views,
3. reducing the number of plotted 3D points when necessary,
4. displaying the results in a single matplotlib figure.

---

## Main Functions

This file contains two main functions:

1. `_sample_points(...)`
2. `main(...)`

It also contains a command-line interface at the bottom.

---

## High-Level Role in the Pipeline

At a high level, this file works like this:

```text
1. Generate a 3D synthetic mask and volume
2. Create 2D projections of the mask
3. Create a montage of selected z-slices
4. Create a 3D scatter plot of mask voxels
5. Show all views together for inspection
```

So this file is a visual inspection tool for the 3D geometry generation stage.

---

## Inputs

### `_sample_points(points, max_points=20000, seed=0)`

#### Inputs
- `points`  
  array of 3D points

- `max_points`  
  maximum number of points to keep

- `seed`  
  random seed for reproducible sampling

#### Output
- original points if already small enough,
- or a random downsampled subset of points

---

### `main(...)`

The main function accepts parameters for both structure models and visualization.

#### Shared inputs
- `shape`
- `seed`
- `max_points`
- `model`

#### Model A inputs
- `n_tubes`
- `tube_radius`

#### Model B inputs
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

## Outputs

The script does not save files by default.

Its main output is an interactive matplotlib figure containing:

1. XY projection
2. XZ projection
3. YZ projection
4. z-slice montage
5. 3D scatter plot
6. text information panel

---

## Function: `_sample_points(points, max_points=20000, seed=0)`

This helper limits the number of points plotted in the 3D scatter view.

### Why it is needed

A full 3D mask may contain many voxels.  
Plotting all of them could make the visualization slow or cluttered.

So this helper randomly downsamples the points if there are too many.

### Pseudocode

```text
FUNCTION _sample_points(points, max_points = 20000, seed = 0)

    IF number of points <= max_points THEN
        RETURN points unchanged
    END IF

    CREATE random number generator with seed

    RANDOMLY choose max_points indices without replacement

    RETURN sampled points
END FUNCTION
```

---

## Main Function: `main(...)`

This is the core function of the visualization script.

### What it does

1. choose which generator to use,
2. generate a 3D volume and mask,
3. convert the mask to boolean,
4. compute orthogonal max projections,
5. select a small set of z-slices,
6. create a matplotlib figure,
7. show several visualization panels,
8. show a 3D scatter plot of mask voxels,
9. display summary information,
10. render the full figure.

---

## Step 1: Generate the structure

Depending on the chosen model:

- Model A calls `generate_mito_volume(...)`
- Model B calls `generate_mito_volume_polyline(...)`

### Pseudocode

```text
IF model is "A" THEN
    CALL generate_mito_volume(...)
    GET volume and mask
ELSE IF model is "B" THEN
    CALL generate_mito_volume_polyline(...)
    GET volume and mask
ELSE
    RAISE error
END IF
```

---

## Step 2: Compute orthogonal projections

The script creates three max projections from the mask:

- `xy`: maximum over Z
- `xz`: maximum over Y
- `yz`: maximum over X

These help the user see the overall structure from different viewpoints.

### Pseudocode

```text
xy = maximum of mask over z axis
xz = maximum of mask over y axis
yz = maximum of mask over x axis
```

---

## Step 3: Select z-slices for montage

The script chooses up to 8 evenly spaced z-slices from the volume.

These are later stitched side by side into one montage image.

### Pseudocode

```text
SELECT evenly spaced z indices from 0 to last z slice
LIMIT number of selected slices to at most 8
```

---

## Step 4: Build the figure layout

The script creates a `2 x 3` figure layout:

### Top row
1. XY projection
2. XZ projection
3. YZ projection

### Bottom row
4. z-slice montage
5. 3D scatter plot
6. information panel

---

## Full Main Function Pseudocode

```text
FUNCTION main(...)

    IF model is "A" THEN
        CALL generate_mito_volume(...)
        GET vol, mask
    ELSE IF model is "B" THEN
        CALL generate_mito_volume_polyline(...)
        GET vol, mask
    ELSE
        RAISE error
    END IF

    CONVERT mask to boolean

    # Orthogonal projections
    xy = maximum projection over Z
    xz = maximum projection over Y
    yz = maximum projection over X

    # Select z-slices for montage
    z_slices = evenly spaced slice indices across z dimension

    CREATE matplotlib figure of size (14, 8)

    # Panel 1: XY projection
    CREATE subplot at position (2, 3, 1)
    SHOW xy image in grayscale
    SET title
    TURN axis off

    # Panel 2: XZ projection
    CREATE subplot at position (2, 3, 2)
    SHOW xz image in grayscale
    SET title
    TURN axis off

    # Panel 3: YZ projection
    CREATE subplot at position (2, 3, 3)
    SHOW yz image in grayscale
    SET title
    TURN axis off

    # Panel 4: z-slice montage
    CREATE subplot at position (2, 3, 4)
    CONCATENATE selected z slices side by side
    SHOW montage in grayscale
    SET title
    TURN axis off

    # Prepare points for 3D scatter
    FIND all foreground voxel coordinates using argwhere(mask)
    DOWNSAMPLE points using _sample_points(...)

    # Panel 5: 3D scatter
    CREATE subplot at position (2, 3, 5) with 3D projection
    PLOT foreground points as scatter:
        x = voxel x coordinate
        y = voxel y coordinate
        z = voxel z coordinate
    SET title
    SET axis labels X, Y, Z

    # Panel 6: information panel
    CREATE subplot at position (2, 3, 6)
    TURN axis off
    DISPLAY text containing:
        volume shape
        number of mask voxels
        n_tubes
        tube_radius
        seed

    APPLY tight layout
    SHOW figure
END FUNCTION
```

---

## Important Visualization Details

### Orthogonal projections

These give a quick summary of the global shape from three different directions.

### z-slice montage

This shows how the structure appears across depth.

### 3D voxel scatter

This provides a more intuitive 3D view of the generated structure.

### Info panel

This gives a quick textual summary of the generation settings.

---

## Command-Line Interface

At the bottom of the file, there is a CLI that lets the script be run directly from the terminal.

### Supported arguments

#### Shared
- `--seed`
- `--model`
- `--max-points`

#### Model A
- `--n-tubes`
- `--tube-radius`

#### Model B
- `--n-seeds`
- `--step-length`
- `--n-steps`
- `--branch-prob`
- `--branch-steps`
- `--radius-min`
- `--radius-max`
- `--persistence`
- `--z-scale`
- `--seed-spread`

---

## CLI Pseudocode

```text
IF file is executed as main program THEN

    CREATE argument parser

    ADD argument --seed
    ADD argument --model
    ADD argument --n-tubes
    ADD argument --tube-radius
    ADD argument --max-points

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

    PARSE command-line arguments

    CALL main(...) using parsed values

END IF
```

---

## Notes

- The script uses the mask, not the intensity volume, for most visualizations.
- The 3D scatter plot uses voxel coordinates from `np.argwhere(mask)`.
- The scatter plot is downsampled when necessary to keep plotting manageable.
- The info panel always shows `n_tubes` and `tube_radius`, even if Model B is used.  
  That means the displayed info is more naturally suited to Model A and could be improved for Model B.
- The script is mainly a debugging and inspection tool, not part of automated training.

---

## Code Observation

One small design observation:

The info panel always shows:

- `n_tubes`
- `tube_radius`

even when `model="B"` is selected.

That is not a runtime bug, but the displayed info is partially Model-A-specific.  
A future improvement would be to show model-specific info depending on whether A or B was used.

---

## Relationship to Other Project Files

This file depends on the structure-generation code.

### Relationship summary

- `shapes.py`
  creates the 3D structures

- `viz_3d.py`
  visualizes those structures in multiple ways

This file is especially useful before moving on to:

- `psf.py`
- `render.py`
- `dataset_generator.py`

because it lets the user confirm that the raw 3D structure itself looks reasonable.

---

## Why this file is important in the thesis project

This file is important because it allows visual inspection of the generated 3D shapes before the rest of the image-formation process is applied.

That is valuable for both:

- debugging the generation logic,
- explaining the pipeline in the thesis.

It helps answer questions like:

- Do the generated tubules look plausible?
- Does Model B produce more realistic 3D geometry than Model A?
- Are the structures too sparse, too thick, or too fragmented?

In simple terms:

- `shapes.py` builds the 3D objects,
- `viz_3d.py` lets you inspect those objects directly.
