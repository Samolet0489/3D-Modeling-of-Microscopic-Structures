# metrics.py

## Purpose

This file defines the segmentation metrics used for evaluating 2D image segmentation results.

Its purpose is to provide simple utility functions for:

- converting raw model logits into binary segmentation masks,
- computing the Dice coefficient,
- computing the IoU score,
- testing those metric functions with small example tensors.

This file is part of the evaluation side of the project.  
It does not generate data or train models.  
Instead, it measures how well predicted segmentation masks match the target masks.

---

## Main Functions

This file contains three main metric-related functions:

1. `threshold_logits(...)`
2. `dice_coeff(...)`
3. `iou_score(...)`

It also contains a small `__main__` test block at the bottom.

---

## Inputs

### 1. `threshold_logits(...)`

#### Inputs
- `logits`  
  A tensor of shape `(B, 1, H, W)` containing raw model outputs.

- `thr`  
  Threshold value used after sigmoid conversion. Default is `0.5`.

#### Output
- A binary prediction tensor of shape `(B, 1, H, W)` with values `{0, 1}`.

---

### 2. `dice_coeff(...)`

#### Inputs
- `pred`  
  Binary predicted mask tensor of shape `(B, 1, H, W)`.

- `target`  
  Binary target mask tensor of shape `(B, 1, H, W)`.

- `eps`  
  Small epsilon for numerical stability.

#### Output
- A scalar tensor representing the Dice coefficient.

---

### 3. `iou_score(...)`

#### Inputs
- `pred`  
  Binary predicted mask tensor of shape `(B, 1, H, W)`.

- `target`  
  Binary target mask tensor of shape `(B, 1, H, W)`.

- `eps`  
  Small epsilon for numerical stability.

#### Output
- A scalar tensor representing the IoU score.

---

## High-Level Logic

At a high level, the file works like this:

1. Convert model logits into binary segmentation predictions.
2. Compare predicted masks with target masks.
3. Compute overlap-based segmentation metrics.
4. Optionally run a built-in test routine to verify correctness.

So this file is a small but important evaluation helper in the project.

---

## Function: `threshold_logits(logits, thr=0.5)`

This function converts raw model logits into binary predictions.

### What it does

- applies a sigmoid activation to transform logits into probabilities,
- applies a threshold,
- returns a binary tensor.

### Formula idea

```text
probability = sigmoid(logit)
prediction = 1 if probability > threshold else 0
```

### Pseudocode

```text
FUNCTION threshold_logits(logits, thr = 0.5)

    APPLY sigmoid to logits
    STORE result as probabilities

    COMPARE probabilities with threshold
    SET values > threshold to 1
    SET other values to 0

    CONVERT result to float tensor

    RETURN binary predictions
END FUNCTION
```

---

## Function: `dice_coeff(pred, target, eps=1e-6)`

This function computes the Dice coefficient.

The Dice coefficient is also often interpreted as the F1 score for segmentation.

### Dice formula

```text
Dice = 2 * |X ∩ Y| / (|X| + |Y|)
```

where:

- `X` = predicted foreground
- `Y` = target foreground

### What it does

1. Flatten prediction and target tensors.
2. Compute intersection.
3. Compute total foreground size.
4. Apply the Dice formula.
5. Return the result.

### Pseudocode

```text
FUNCTION dice_coeff(pred, target, eps = 1e-6)

    FLATTEN pred into 1D tensor
    FLATTEN target into 1D tensor

    COMPUTE intersection = sum(pred * target)

    COMPUTE dice =
        (2 * intersection + eps) /
        (sum(pred) + sum(target) + eps)

    RETURN dice
END FUNCTION
```

---

## Function: `iou_score(pred, target, eps=1e-6)`

This function computes Intersection over Union (IoU), also called the Jaccard index.

### IoU formula

```text
IoU = |X ∩ Y| / |X ∪ Y|
```

where:

- `X` = predicted foreground
- `Y` = target foreground

### What it does

1. Flatten prediction and target tensors.
2. Compute intersection.
3. Compute union.
4. Apply the IoU formula.
5. Return the result.

### Pseudocode

```text
FUNCTION iou_score(pred, target, eps = 1e-6)

    FLATTEN pred into 1D tensor
    FLATTEN target into 1D tensor

    COMPUTE intersection = sum(pred * target)

    COMPUTE union =
        sum(pred + target - pred * target)

    COMPUTE iou =
        (intersection + eps) /
        (union + eps)

    RETURN iou
END FUNCTION
```

---

## Why flattening is used

Both `dice_coeff(...)` and `iou_score(...)` flatten the tensors before computing overlap.

This is done so that all pixels across the batch and image dimensions are treated as one long vector.

In other words, the metric is computed globally over the whole tensor.

Conceptually:

```text
(B, 1, H, W)  ->  one long 1D array
```

This makes the implementation simple and compact.

---

## `__main__` Test Block

At the bottom of the file, there is a built-in test routine.

This routine checks whether the metric functions behave as expected.

### What it does

1. Create random logits.
2. Create a random target mask.
3. Create a perfect prediction.
4. Create a random prediction.
5. Test:
   - thresholding
   - Dice score
   - IoU score
6. If CUDA is available, repeat some tests on GPU.

---

## Pseudocode for Test Block

```text
IF file is executed as main program THEN

    PRINT test start message

    SET batch_size = 2
    SET image height and width = 64, 64

    CREATE random logits tensor

    CREATE random binary target tensor

    SET pred_perfect = copy of target
    CREATE pred_random as random binary tensor

    PRINT shapes of logits and target

    TEST threshold_logits:
        CALL threshold_logits(logits, thr = 0.5)
        PRINT shape and unique values

    TEST dice_coeff:
        COMPUTE Dice for perfect prediction
        COMPUTE Dice for random prediction
        PRINT both values

    TEST iou_score:
        COMPUTE IoU for perfect prediction
        COMPUTE IoU for random prediction
        PRINT both values

    TEST GPU path:
        IF CUDA is available THEN
            MOVE tensors to GPU
            CALL threshold_logits on GPU
            COMPUTE Dice and IoU on GPU
            PRINT GPU results
        ELSE
            PRINT that CUDA is not available
        END IF

    PRINT test completed message

END IF
```

---

## Expected Metric Behavior

These functions should behave in an intuitive way.

### Perfect prediction
If `pred` is exactly equal to `target`:

- Dice should be very close to `1.0`
- IoU should be very close to `1.0`

### Poor prediction
If `pred` differs strongly from `target`:

- Dice should be lower
- IoU should be lower

### Random prediction
For random masks:

- Dice will usually be somewhere in the middle or low
- IoU will usually also be lower than a perfect prediction

---

## Notes

- The functions assume **binary segmentation masks**.
- `threshold_logits(...)` expects **raw logits**, not already thresholded masks.
- `dice_coeff(...)` and `iou_score(...)` expect binary inputs.
- `eps` is included to avoid division-by-zero problems.
- The metrics are implemented in a simple global form, not per-image and then averaged.
- The file is small, but it is central for evaluating segmentation quality.

---

## Relationship to the training pipeline

This file is used after the model produces segmentation outputs.

Typical workflow:

```text
1. Model outputs logits
2. threshold_logits(...) converts logits to binary mask
3. dice_coeff(...) measures overlap with target
4. iou_score(...) measures overlap with target
5. training / validation code reports these values
```

So this file fits naturally into validation and testing.

---

## Why this file is important in the project

This file is important because it defines how segmentation performance is measured.

In the thesis project, these metrics are necessary for answering whether the synthetic-data approach improves model performance.

Without this file, the project could still generate images and train models, but it would not have a clean way to quantify segmentation quality.

In simple terms:

- `dataset_generator.py` creates the data,
- `dataset2d.py` loads the data,
- the model predicts segmentation masks,
- `metrics.py` tells you how good those predictions are.
