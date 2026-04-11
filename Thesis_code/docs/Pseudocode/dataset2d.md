# dataset2d.py

## Purpose

This file defines the 2D segmentation dataset and dataloader utilities used for model training.

Its main purpose is to:

- load paired 2D images and masks from disk,
- optionally apply simple paired augmentations,
- convert data into PyTorch tensors,
- split the dataset into training, validation, and test subsets,
- create PyTorch dataloaders for those subsets.

This file is part of the training side of the project.  
While `dataset_generator.py` creates the synthetic dataset, `dataset2d.py` is responsible for reading that dataset back into memory in a form suitable for neural network training.

---

## Expected Folder Structure

The dataset is expected to have the following structure:

```text
data_dir/
├─ images/
│  └─ *.png
└─ masks/
   └─ *.png
```

Important rules:

- images must be grayscale PNG files,
- masks must be grayscale PNG files,
- filenames must match exactly between `images/` and `masks/`.

Example:

```text
images/000001.png
masks/000001.png
```

---

## Main Responsibilities

This file has two main responsibilities:

### 1. Dataset loading
Handled by the `SegmentationDataset2D` class.

### 2. Dataloader creation and dataset splitting
Handled by the `make_dataloaders(...)` function.

---

## Inputs

### Dataset class inputs

#### `data_dir`
Root folder containing `images/` and `masks/`.

#### `augment`
Boolean flag that decides whether random augmentations are applied.

#### `seed`
Random seed used for reproducibility.

---

### Dataloader function inputs

#### `data_dir`
Root dataset folder.

#### `batch_size`
Batch size for PyTorch dataloaders.

#### `val_split`
Fraction of the dataset used for validation.

#### `test_split`
Fraction of the dataset used for testing.

#### `seed`
Random seed for deterministic splitting.

#### `num_workers`
Number of worker processes for the dataloaders.

#### `pin_memory`
Whether PyTorch should pin memory for faster GPU transfer.

---

## Outputs

### Dataset class outputs

Each call to `__getitem__(idx)` returns:

- `image_tensor` of shape `(1, H, W)` with values in `[0, 1]`
- `mask_tensor` of shape `(1, H, W)` with values in `{0, 1}`

Both are `torch.Tensor` objects.

---

### Dataloader function outputs

`make_dataloaders(...)` returns:

- `train_loader`
- `val_loader`
- `test_loader`

These are PyTorch `DataLoader` objects.

---

## High-Level Logic

At a high level, the file works like this:

1. Read the dataset folder.
2. Collect all image filenames.
3. Check that matching mask files exist.
4. When a sample is requested:
   - load image,
   - load mask,
   - normalize image,
   - binarize mask,
   - optionally apply paired augmentations,
   - convert both to tensors.
5. For training:
   - create deterministic train/validation/test splits,
   - build separate datasets for train and evaluation,
   - wrap them in dataloaders.

---

## Main Class: `SegmentationDataset2D`

This class inherits from `torch.utils.data.Dataset`.

It represents the dataset as a PyTorch-compatible object.

---

### `__init__(self, data_dir, augment=True, seed=0)`

This method initializes the dataset object.

### What it does

- stores configuration values,
- defines `images/` and `masks/` folder paths,
- reads all PNG image filenames,
- checks that the dataset is not empty,
- checks that every image has a matching mask.

### Pseudocode

```text
FUNCTION __init__(data_dir, augment = True, seed = 0)

    STORE data_dir as Path object
    STORE augment flag
    STORE seed

    SET images_dir = data_dir / "images"
    SET masks_dir  = data_dir / "masks"

    READ all PNG filenames from images_dir
    SORT filenames

    IF no image files are found THEN
        RAISE error
    END IF

    STORE filenames

    FOR each filename
        BUILD corresponding mask path
        IF mask file does not exist THEN
            RAISE error
        END IF
    END FOR

END FUNCTION
```

---

### `__len__(self)`

Returns the number of samples in the dataset.

### Pseudocode

```text
FUNCTION __len__()
    RETURN number of filenames
END FUNCTION
```

---

### `__getitem__(self, idx)`

Loads one image-mask pair and returns it as tensors.

### What it does

1. Find the filename for the requested index.
2. Load image and mask from disk.
3. Convert both to grayscale.
4. Convert them into NumPy arrays.
5. Normalize the image from `[0, 255]` to `[0, 1]`.
6. Convert the mask into binary `{0, 1}`.
7. Optionally apply augmentations.
8. Ensure arrays are contiguous in memory.
9. Convert arrays into PyTorch tensors.
10. Add a channel dimension.

### Important note

The current code creates tensors twice in a row:

```python
image_tensor = torch.from_numpy(image_np).unsqueeze(0)
mask_tensor  = torch.from_numpy(mask_np).unsqueeze(0)

image_tensor = torch.from_numpy(image_np).unsqueeze(0)
mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
```

This is redundant.  
It does not change the result, but it is unnecessary duplication in the code.

### Pseudocode

```text
FUNCTION __getitem__(idx)

    GET filename at position idx

    BUILD image_path
    BUILD mask_path

    LOAD image as grayscale
    LOAD mask as grayscale

    CONVERT image to float32 NumPy array
    CONVERT mask to float32 NumPy array

    NORMALIZE image to [0, 1]

    CONVERT mask to binary:
        values > 127 become 1
        other values become 0

    IF augmentations are enabled THEN
        APPLY paired augmentations to image and mask
    END IF

    MAKE image array contiguous
    MAKE mask array contiguous

    CONVERT image array to torch tensor
    ADD channel dimension

    CONVERT mask array to torch tensor
    ADD channel dimension

    RETURN image_tensor, mask_tensor
END FUNCTION
```

---

## Helper Method: `_apply_augmentations(image, mask)`

This method applies paired augmentations.

Paired means that spatial changes must be applied to both image and mask in the same way, so they still align.

### Current augmentations

- random horizontal flip
- random vertical flip
- random brightness adjustment on image only

### Pseudocode

```text
FUNCTION _apply_augmentations(image, mask)

    IF random number < 0.5 THEN
        FLIP image left-right
        FLIP mask left-right
    END IF

    IF random number < 0.5 THEN
        FLIP image up-down
        FLIP mask up-down
    END IF

    IF random number < 0.3 THEN
        SAMPLE brightness factor between 0.9 and 1.1
        MULTIPLY image by brightness factor
        CLIP image to [0, 1]
    END IF

    RETURN image, mask
END FUNCTION
```

---

## Main Function: `make_dataloaders(...)`

This function creates train, validation, and test dataloaders.

### Why this function is important

It performs deterministic dataset splitting and avoids a common problem:

- training data should use augmentations,
- validation and test data should not use augmentations.

To solve this cleanly, the code creates:

- one base dataset with `augment=True` for training,
- one base dataset with `augment=False` for validation and testing.

Then it uses subsets of those datasets with the same deterministic split indices.

### What it does

1. Seed random number generators.
2. Create a base dataset to determine total size.
3. Compute train/validation/test split sizes.
4. Create a deterministic shuffled list of indices.
5. Split those indices into train, val, and test.
6. Create separate training and evaluation dataset objects.
7. Build subsets from those datasets.
8. Wrap subsets in `DataLoader` objects.

### Pseudocode

```text
FUNCTION make_dataloaders(
    data_dir,
    batch_size,
    val_split = 0.1,
    test_split = 0.1,
    seed = 0,
    num_workers = 2,
    pin_memory = True
)

    SET torch random seed
    SET numpy random seed

    CREATE base_dataset with augment = False
    GET total dataset size

    IF total size is zero THEN
        RAISE error
    END IF

    COMPUTE test_size
    COMPUTE val_size
    COMPUTE train_size

    CREATE deterministic random permutation of dataset indices

    SPLIT indices into:
        train_indices
        val_indices
        test_indices

    IMPORT Subset

    CREATE train_base dataset with augment = True
    CREATE eval_base dataset with augment = False

    CREATE train_dataset as Subset(train_base, train_indices)
    CREATE val_dataset   as Subset(eval_base, val_indices)
    CREATE test_dataset  as Subset(eval_base, test_indices)

    CREATE train_loader with shuffle = True
    CREATE val_loader with shuffle = False
    CREATE test_loader with shuffle = False

    RETURN train_loader, val_loader, test_loader
END FUNCTION
```

---

## `__main__` Test Block

At the bottom of the file, there is a simple test routine.

### What it does

1. Print a test message.
2. Check whether `./test_data` exists.
3. If it does not exist, skip the test.
4. If it does exist:
   - build train/validation/test loaders,
   - print number of batches,
   - load one batch from the training loader,
   - print shape, dtype, min/max values, and mask values.

### Pseudocode

```text
IF file is executed as main program THEN

    PRINT test start message

    SET test_dir = "./test_data"

    IF test_dir does not exist THEN
        PRINT skip message
    ELSE
        CALL make_dataloaders(...) using test_dir

        PRINT number of batches in:
            train_loader
            val_loader
            test_loader

        GET first batch from train_loader

        PRINT:
            image tensor shape
            image dtype
            image min and max
            mask tensor shape
            mask dtype
            unique mask values

        PRINT success message
    END IF

END IF
```

---

## Notes

- The dataset is designed for **binary segmentation**.
- Images are treated as **single-channel grayscale**.
- Masks are converted into binary values using threshold `> 127`.
- Images are normalized into `[0, 1]`.
- Masks remain binary `{0, 1}`.
- Augmentations are intentionally simple and minimal.
- Horizontal and vertical flips are paired.
- Brightness augmentation is applied only to the image, not the mask.
- The split is deterministic because it uses a seeded permutation.

---

## Code Observation

There is one small code issue worth documenting:

### Redundant tensor creation
Inside `__getitem__`, tensors are created twice one after the other.

That section can be simplified to only one creation block.

So conceptually the intended logic is:

```text
MAKE arrays contiguous
CONVERT to tensors once
RETURN tensors
```

That is a cleanup issue, not a logic-breaking issue.

---

## Why this file is important in the project

This file is important because it bridges the generated dataset and the neural network training pipeline.

Its role is:

- to read saved synthetic datasets,
- to prepare them for PyTorch,
- to ensure images and masks stay aligned,
- to apply safe training augmentations,
- to provide reproducible train/validation/test splits.

In simple terms:

- `dataset_generator.py` creates the dataset,
- `dataset2d.py` loads the dataset for training,
- training code depends on this file to feed correct batches into the model.

So this file is a key part of the full training workflow.
