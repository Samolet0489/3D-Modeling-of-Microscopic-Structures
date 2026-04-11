# shapes.py

## Purpose

This file contains the 3D shape-generation utilities used in the synthetic microscopy pipeline.

Its purpose is to create synthetic 3D biological-like structures that later become microscopy images.

It supports two main kinds of structure generation:

- a simple ellipsoid volume,
- mitochondria-like tubular volumes.

The file includes:

- basic geometric generation,
- a simple tube-based baseline model,
- a more advanced polyline / random-walk model,
- several helper functions for safe movement, rasterization, dilation, and border handling.

This file is one of the core files of the whole project because it defines the actual synthetic structures before they are blurred, rendered, and noised.

---

## Main Responsibilities

This file is responsible for:

1. creating 3D binary masks for synthetic structures,
2. creating matching intensity volumes,
3. supporting both simple and more advanced mitochondria-like geometry,
4. handling random-walk style tubular growth safely inside the 3D volume.

---

## Main Functions

This file contains the following main public functions:

1. `create_ellipsoid_volume(...)`
2. `generate_mito_volume(...)`
3. `generate_mito_volume_polyline(...)`

It also contains several helper functions used by the polyline generator.

---

## High-Level Role in the Pipeline

At a high level, this file is used like this:

```text
1. Create a synthetic 3D structure
2. Return:
   - intensity volume
   - binary mask
3. Pass the volume to PSF blur and rendering
4. Pass the mask to projection / mask generation
```

So this file is the beginning of the image-generation pipeline.

---

## Inputs

The main inputs across the functions are:

- `shape` or `volume_shape`  
  size of the 3D volume in `(Z, Y, X)`

- `center`
- `radii`
- `intensity`

- `n_tubes`
- `tube_radius`

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

- `seed`

---

## Outputs

The main public functions return:

- `volume`  
  a float32 3D intensity volume

- `mask`  
  a boolean 3D foreground mask

So each generated structure has both:

- a geometry mask,
- an intensity representation.

---

## Function: `create_ellipsoid_volume(...)`

This function creates a simple solid ellipsoid inside a 3D volume.

### What it does

1. define the volume size,
2. choose a center,
3. evaluate the ellipsoid equation at every voxel,
4. create a binary mask where the ellipsoid exists,
5. create an intensity volume using that mask.

### Mathematical idea

A voxel belongs to the ellipsoid if:

```text
((z - z0) / rz)^2 + ((y - y0) / ry)^2 + ((x - x0) / rx)^2 <= 1
```

### Pseudocode

```text
FUNCTION create_ellipsoid_volume(
    volume_shape = (32, 128, 128),
    center = None,
    radii = (6, 20, 20),
    intensity = 1.0
)

    EXTRACT z_size, y_size, x_size from volume_shape

    IF center is None THEN
        SET center to middle of volume
    END IF

    EXTRACT z0, y0, x0 from center
    EXTRACT rz, ry, rx from radii

    CREATE z coordinate grid
    CREATE y coordinate grid
    CREATE x coordinate grid

    COMPUTE ellipsoid equation for all voxels

    CREATE binary mask where ellipsoid equation <= 1

    CREATE zero volume
    SET volume inside mask = intensity

    RETURN volume, mask
END FUNCTION
```

---

## Function: `generate_mito_volume(...)`

This function creates a very simple baseline mitochondria-like volume made of straight cylindrical tubes.

### What it does

- randomly places several circular tube cross-sections in the XY plane,
- extends them through the full Z dimension,
- creates a simple vertical-tube structure.

This is the simpler baseline model.

### Pseudocode

```text
FUNCTION generate_mito_volume(
    shape = (32, 128, 128),
    n_tubes = 5,
    tube_radius = 2,
    intensity = 1.0,
    seed = None
)

    ROUND tube_radius up to integer

    EXTRACT z_size, y_size, x_size from shape
    CREATE random generator

    CREATE empty volume
    CREATE empty boolean mask

    CREATE y grid
    CREATE x grid

    FOR each tube

        RANDOMLY choose tube center cy, cx inside safe image range

        COMPUTE 2D distance from every pixel to tube center

        CREATE circular slice mask where distance <= tube_radius

        SET all z slices at those XY positions to True in mask
        SET all z slices at those XY positions to intensity in volume

    END FOR

    RETURN volume, mask
END FUNCTION
```

---

## Helper Function: `_safe_unit_vector(vec)`

This helper converts a vector into a unit vector safely.

### What it does

- computes vector norm,
- if norm is almost zero, returns a default direction,
- otherwise returns normalized vector.

### Pseudocode

```text
FUNCTION _safe_unit_vector(vec)

    CONVERT vec to float32 array
    COMPUTE norm

    IF norm is very small THEN
        RETURN default direction [0, 1, 0]
    END IF

    RETURN vec / norm
END FUNCTION
```

---

## Helper Function: `_safe_box(shape, margin)`

This helper defines a safe interior box inside the volume.

### What it does

- leaves a margin away from all borders,
- returns lower and upper coordinate bounds.

### Pseudocode

```text
FUNCTION _safe_box(shape, margin)

    EXTRACT z_size, y_size, x_size from shape

    SET low  = [margin, margin, margin]
    SET high = [z_size - margin - 1,
                y_size - margin - 1,
                x_size - margin - 1]

    RETURN low, high
END FUNCTION
```

---

## Helper Function: `_clip_position(pos, shape, margin)`

Clips a position so it stays inside the safe box.

```text
FUNCTION _clip_position(pos, shape, margin)

    GET low, high from _safe_box(shape, margin)

    COPY position
    CLIP position between low and high

    RETURN clipped position
END FUNCTION
```

---

## Helper Function: `_rasterize_segment(start, end, oversample=3.0)`

Returns interpolated points along a line segment.

### What it does

- computes the segment length,
- decides how many interpolation points are needed,
- returns points from start to end.

This is used so the generated tubular path does not have gaps between steps.

### Pseudocode

```text
FUNCTION _rasterize_segment(start, end, oversample = 3.0)

    CONVERT start and end to float32 arrays

    COMPUTE segment vector
    COMPUTE segment distance

    COMPUTE number of interpolation points from distance and oversampling
    CREATE interpolation parameter values from 0 to 1

    RETURN list of interpolated points along the segment
END FUNCTION
```

---

## Helper Function: `_make_ball_structure(radius)`

Creates a 3D spherical structuring element.

This is later used for dilation.

### Pseudocode

```text
FUNCTION _make_ball_structure(radius)

    SET r = integer radius

    CREATE local coordinate grids from -r to +r

    RETURN boolean ball where:
        z^2 + y^2 + x^2 <= r^2
END FUNCTION
```

---

## Helper Function: `_border_repulsion(pos, shape, margin, band)`

Computes a soft repulsion vector that pushes walkers away from borders.

### What it does

- checks whether a position is entering a warning band near a boundary,
- adds repulsion components that point back inward.

This helps prevent walkers from constantly hitting the edges.

### Pseudocode

```text
FUNCTION _border_repulsion(pos, shape, margin, band)

    GET safe box bounds

    CREATE zero repulsion vector

    FOR each axis

        COMPUTE low warning band
        COMPUTE high warning band

        IF position is too close to low side THEN
            ADD positive repulsion on that axis
        ELSE IF position is too close to high side THEN
            ADD negative repulsion on that axis
        END IF

    END FOR

    RETURN repulsion vector
END FUNCTION
```

---

## Helper Function: `_turn_inward_from_bounds(...)`

Corrects a walker direction if the proposed move would leave the safe interior box.

### What it does

- checks whether the proposed next position would cross the safe boundary,
- reflects the offending direction components,
- biases the corrected direction back toward the colony center,
- returns corrected direction and a boundary-hit flag.

### Pseudocode

```text
FUNCTION _turn_inward_from_bounds(prev_pos, proposed_dir, shape, margin, colony_center)

    GET safe box bounds

    COPY proposed direction
    COMPUTE proposed position = prev_pos + proposed_dir

    SET hit_boundary = False

    FOR each axis
        IF proposed position is below low bound THEN
            REFLECT direction component to positive
            hit_boundary = True
        ELSE IF proposed position is above high bound THEN
            REFLECT direction component to negative
            hit_boundary = True
        END IF
    END FOR

    IF boundary was hit THEN
        COMPUTE inward direction toward colony center
        NORMALIZE inward vector
        MIX reflected direction with inward direction
        NORMALIZE result
    END IF

    RETURN corrected direction, hit_boundary
END FUNCTION
```

---

## Main Function: `generate_mito_volume_polyline(...)`

This is the more advanced mitochondria-like generator.

It creates smoother tubule networks using persistent random walks and optional branching.

This is the more realistic structure model in the file.

---

### Main idea

The function works by:

1. choosing a colony center,
2. placing several seed walkers near that center,
3. letting each walker move through 3D space,
4. giving each walker direction persistence,
5. allowing some branching,
6. rasterizing the walked paths,
7. dilating those centerlines into tubes,
8. producing the final volume and mask.

---

### Important concepts

#### Persistent random walk
Each walker does not choose a completely random direction every time.  
Instead, it keeps some memory of its previous direction.

#### Border safety
The walker is softly repelled from borders and turned inward if it tries to leave the safe region.

#### Branching
Some walkers can spawn one branch walker.

#### Radius variation
Walker tube radius changes slightly over time, within limits.

#### Dilation
Centerline points are expanded into solid tubes using a spherical structuring element.

---

### Step-by-step logic

#### 1. Parameter preparation
- round radii upward,
- validate radius bounds,
- clip persistence to valid range.

#### 2. Colony center creation
- choose a safe colony center,
- use one z center and one XY center.

#### 3. Seed walker creation
- create several starting positions around the colony center,
- assign random directions,
- assign initial radii,
- set available steps and branching ability.

#### 4. Random-walk growth loop
For each active walker:
- compute noise,
- compute border repulsion,
- combine persistence, random noise, and border effects,
- reduce z motion using `z_scale`,
- correct direction if a boundary would be crossed,
- terminate walkers that hit the border too often,
- move walker forward,
- slightly update radius,
- rasterize the segment between old and new position,
- store centerline points with radius,
- optionally create one new branch walker.

#### 5. Radius-grouped dilation
- group all stored points by radius,
- create a binary centerline volume for each radius,
- dilate that volume with a 3D ball structuring element,
- combine all dilated tubes into the final mask.

#### 6. Intensity volume creation
- create zero volume,
- set mask voxels to the requested intensity.

#### 7. Return final volume and mask

---

### Pseudocode

```text
FUNCTION generate_mito_volume_polyline(
    shape = (32, 128, 128),
    n_seeds = 4,
    step_length = 1,
    n_steps = 80,
    branch_prob = 0.08,
    branch_steps = 40,
    radius_min = 1,
    radius_max = 3,
    persistence = 0.85,
    z_scale = 0.35,
    seed_spread = 20,
    intensity = 1.0,
    seed = None
)

    ROUND radius_min and radius_max upward

    IF radius_min > radius_max THEN
        RAISE error
    END IF

    CLIP persistence into [0, 1]

    EXTRACT z_size, y_size, x_size from shape
    CREATE random number generator

    COMPUTE safe margin

    SAMPLE safe colony z coordinate
    SAMPLE safe colony y coordinate
    SAMPLE safe colony x coordinate

    BUILD colony_center vector

    COMPUTE z spread for seeds
    COMPUTE turn band for border repulsion

    CREATE empty list of active walkers

    FOR each seed walker
        SAMPLE start position around colony center
        CLIP start position to safe box

        SAMPLE random initial direction
        NORMALIZE direction

        SAMPLE initial radius

        CREATE walker dictionary with:
            position
            direction
            remaining steps
            branching ability
            radius
            edge hit counter

        ADD walker to active walkers
    END FOR

    CREATE dictionary point_to_radius

    WHILE active walkers exist

        CREATE empty list new_walkers
        CREATE empty list survivors

        FOR each walker

            IF walker has no steps left THEN
                SKIP walker
            END IF

            STORE previous position

            SAMPLE random directional noise

            COMPUTE border repulsion
            reduce z repulsion slightly

            MIX:
                persistence * old direction
                + random noise contribution
                + border repulsion contribution

            SCALE z component by z_scale
            NORMALIZE direction

            COMPUTE step vector

            CORRECT direction using _turn_inward_from_bounds(...)

            UPDATE edge hit count

            IF edge hit count too large THEN
                DISCARD walker
                CONTINUE
            END IF

            NORMALIZE corrected direction

            COMPUTE new position
            CLIP new position to safe box

            UPDATE walker radius with small random drift
            CLIP radius to valid range
            ROUND radius to integer for rasterization

            FOR each interpolated point from previous position to new position
                ROUND point to voxel coordinates
                IF point is inside volume THEN
                    STORE point with maximum encountered radius
                END IF
            END FOR

            UPDATE walker position
            UPDATE walker direction
            DECREASE steps left

            IF walker still has steps left THEN
                ADD walker to survivors
            END IF

            IF walker can still spawn AND did not recently hit border AND random branch event occurs THEN
                SAMPLE branch step count
                SAMPLE branch direction near current direction
                SAMPLE branch radius near current radius

                CREATE new branch walker
                ADD it to new_walkers
            END IF

        END FOR

        SET active_walkers = survivors + new_walkers

    END WHILE

    CREATE dictionary radius_to_points for each radius

    FOR each stored centerline point
        GROUP point by radius
    END FOR

    CREATE empty final boolean mask

    FOR each radius group

        IF no points exist for this radius THEN
            CONTINUE
        END IF

        CREATE empty boolean centerline volume
        MARK all stored points for this radius

        CREATE ball structuring element for this radius

        DILATE centerline volume with the ball

        OR the result into final mask
    END FOR

    CREATE zero intensity volume
    SET intensity inside final mask

    RETURN volume, mask
END FUNCTION
```

---

## Relationship Between the Public Generators

### `create_ellipsoid_volume(...)`
A simple geometric primitive.

### `generate_mito_volume(...)`
A very simple baseline model with straight full-depth tubes.

### `generate_mito_volume_polyline(...)`
A more advanced, more realistic tubular-network generator.

So the file supports a progression from:

```text
simple shape
→ simple tubes
→ more realistic branched tubules
```

---

## Relationship to Other Project Files

This file provides the synthetic structures that are used by the rest of the image pipeline.

### Relationship summary

- `shapes.py`  
  creates the synthetic 3D structures

- `psf.py`  
  blurs those structures with an optical model

- `render.py`  
  projects them into 2D images

- `noise.py`  
  adds realistic imaging artifacts

- `dataset_generator.py`  
  calls `generate_mito_volume(...)` and `generate_mito_volume_polyline(...)`

So `shapes.py` is the structural foundation of the synthetic-data system.

---

## Notes

- All main volumes are created in `(Z, Y, X)` order.
- Output masks are boolean.
- Output intensity volumes are float32.
- The polyline model uses dilation to thicken centerlines into tubes.
- Border handling is important so walkers stay inside a safe region.
- `z_scale` reduces motion in the axial direction, which can help shape realism.
- `persistence` controls how smooth the walker path is.
- `branch_prob` controls how often branches appear.
- Radius changes are small and gradual.

---

## Why this file is important in the thesis project

This file is one of the most important files in the whole thesis codebase because it defines the synthetic structures that everything else depends on.

The quality of the later microscopy images depends heavily on whether the generated shapes are believable.

This file is where the synthetic biological geometry is created.

Without this file:

- there would be no 3D structures,
- no masks,
- no meaningful microscopy simulation.

In simple terms:

- `shapes.py` creates the objects,
- the rest of the pipeline turns those objects into images.

So `shapes.py` is the structural starting point of the entire synthetic-data workflow.
