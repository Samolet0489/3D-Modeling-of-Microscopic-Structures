# train_2d.py

## Purpose

This file is the main training script for 2D U-Net segmentation.

Its purpose is to:

- parse command-line training arguments,
- choose the training device,
- load the dataset and create dataloaders,
- create the 2D U-Net model,
- define the loss and optimizer,
- train the model for multiple epochs,
- validate the model after each epoch,
- save checkpoints,
- save prediction snapshots,
- evaluate the best checkpoint on the test set.

This file is one of the most important parts of the training workflow because it connects:

- the generated dataset,
- the segmentation model,
- the loss function,
- the metric computation,
- the checkpointing and evaluation process.

---

## Main Responsibilities

This file is responsible for:

1. command-line argument parsing,
2. reproducibility setup,
3. device selection,
4. dataset loading and train/validation/test splitting,
5. optional overfit/debug mode with limited training samples,
6. model creation,
7. training and validation loops,
8. checkpoint saving,
9. prediction snapshot creation,
10. final test evaluation.

---

## Main Dependencies

This file uses:

- `SegmentationDataset2D`
- `make_dataloaders`
- `UNet2D`
- `threshold_logits`
- `dice_coeff`
- `iou_score`

It also uses PyTorch utilities such as:

- `torch`
- `torch.nn`
- `torch.optim`
- `torch.amp.autocast`
- `torch.amp.GradScaler`
- `DataLoader`

and utility libraries such as:

- `argparse`
- `random`
- `numpy`
- `PIL.Image`
- `pathlib.Path`

---

## High-Level Role in the Pipeline

At a high level, this file works like this:

```text
1. Read training arguments
2. Set random seeds
3. Choose CPU or GPU
4. Load dataset and create dataloaders
5. Build U-Net model
6. Define loss and optimizer
7. Train for multiple epochs
8. Validate after each epoch
9. Save checkpoints and prediction snapshots
10. Load best checkpoint
11. Evaluate on test set
```

So this file is the central training-and-evaluation script for the 2D segmentation phase.

---

## Inputs

The file accepts many command-line arguments.

They can be grouped into the following categories.

---

### 1. Dataset and training duration

- `--data`
- `--epochs`
- `--batch-size`

---

### 2. Optimization settings

- `--grad-clip`
- `--lr`
- `--pos-weight`

---

### 3. Reproducibility and device

- `--seed`
- `--device`
- `--amp`

---

### 4. Output settings

- `--out-dir`
- `--save-every`

---

### 5. Debug / overfit mode

- `--limit-train`

---

### 6. Prediction threshold

- `--thr`

---

### 7. Dataloader settings

- `--num-workers`

---

## Outputs

This script produces several kinds of outputs.

### 1. Console logs
It prints:

- device info,
- dataset information,
- model size,
- estimated class weighting,
- epoch-by-epoch training and validation results,
- final test metrics.

### 2. Checkpoints
Saved to the output directory:

- `checkpoint_last.pt`
- `checkpoint_best.pt`

### 3. Prediction snapshots
Saved to the output directory:

- `preds_epoch_XXX.png`
- `preds_test.png`

### 4. Final evaluation
At the end, the script evaluates the best checkpoint on the test set.

---

## Function: `parse_args()`

This function defines and parses all command-line arguments.

### What it does

- creates an argument parser,
- adds all supported training arguments,
- parses them,
- returns the parsed argument object.

### Pseudocode

```text
FUNCTION parse_args()

    CREATE argument parser with description

    ADD argument --data
    ADD argument --epochs
    ADD argument --batch-size
    ADD argument --grad-clip
    ADD argument --lr
    ADD argument --seed
    ADD argument --device
    ADD argument --out-dir
    ADD argument --limit-train
    ADD argument --amp
    ADD argument --save-every
    ADD argument --pos-weight
    ADD argument --thr
    ADD argument --num-workers

    RETURN parsed arguments
END FUNCTION
```

---

## Function: `get_device(device_arg)`

This function chooses the computation device.

### What it does

- if `device_arg == "auto"`, choose CUDA if available, otherwise CPU,
- otherwise use the requested device directly.

### Pseudocode

```text
FUNCTION get_device(device_arg)

    IF device_arg is "auto" THEN
        IF CUDA is available THEN
            device = CUDA
        ELSE
            device = CPU
        END IF
    ELSE
        device = requested device
    END IF

    RETURN device
END FUNCTION
```

---

## Function: `print_device_info(device)`

This function prints information about the selected device.

### What it does

- print the device type,
- if CUDA is used, print GPU name and CUDA version.

### Pseudocode

```text
FUNCTION print_device_info(device)

    PRINT selected device

    IF device is CUDA THEN
        GET device index
        TRY to get GPU name
        IF that fails THEN
            USE fallback name
        END IF

        PRINT GPU name
        PRINT CUDA version
    END IF
END FUNCTION
```

---

## Function: `create_pred_snapshot(...)`

This function creates a visualization image showing model predictions.

### What it does

1. switch model to evaluation mode,
2. collect several samples from the validation loader,
3. run the model to get predictions,
4. threshold predictions,
5. create one large grayscale image grid,
6. paste:
   - input image,
   - target mask,
   - predicted mask
   side by side for each sample,
7. save the final visualization.

### Grid layout

Each row contains:

```text
[ input image ] [ target mask ] [ predicted mask ]
```

### Pseudocode

```text
FUNCTION create_pred_snapshot(
    model,
    val_loader,
    device,
    epoch = None,
    out_dir = None,
    threshold = 0.5,
    num_samples = 6,
    suffix = None
)

    SET model to evaluation mode

    INITIALIZE empty lists:
        images_list
        targets_list
        preds_list

    INITIALIZE sample counter

    DISABLE gradients

    FOR each batch in validation loader

        MOVE images to device
        MOVE masks to device

        RUN model forward pass
        THRESHOLD logits into binary predictions

        FOR each sample in batch
            IF enough samples already collected THEN
                BREAK
            END IF

            STORE image
            STORE target
            STORE prediction
            INCREASE sample counter
        END FOR

        IF enough samples already collected THEN
            BREAK
        END IF
    END FOR

    CREATE blank grayscale grid image

    FOR each collected sample
        CONVERT input image to uint8
        CONVERT target mask to uint8
        CONVERT predicted mask to uint8

        PASTE input image into grid
        PASTE target mask into grid
        PASTE predicted mask into grid
    END FOR

    DETERMINE output filename
        use suffix if given
        otherwise use epoch-based filename

    SAVE grid image
    PRINT save message
END FUNCTION
```

---

## Function: `train_one_epoch(...)`

This function performs one full training epoch.

### What it does

For each batch:

- move data to the device,
- zero gradients,
- run forward pass,
- compute loss,
- optionally use automatic mixed precision,
- skip non-finite losses,
- backpropagate,
- optionally clip gradients,
- update optimizer,
- accumulate average loss.

### Pseudocode

```text
FUNCTION train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    use_amp = False,
    scaler = None,
    grad_clip = 1.0
)

    SET model to training mode

    SET total_loss = 0
    SET num_batches = 0

    FOR each batch of images and masks

        MOVE images to device
        MOVE masks to device

        ZERO optimizer gradients

        IF use_amp is True THEN
            ENABLE autocast
            RUN model forward pass
            COMPUTE loss
        ELSE
            RUN model forward pass
            COMPUTE loss
        END IF

        IF loss is not finite THEN
            PRINT warning
            SKIP batch
        END IF

        IF use_amp is True THEN
            SCALE loss and backpropagate
        ELSE
            BACKPROPAGATE loss
        END IF

        IF grad clipping is enabled THEN
            CLIP gradients
        END IF

        IF use_amp is True THEN
            STEP optimizer through scaler
            UPDATE scaler
        ELSE
            STEP optimizer normally
        END IF

        ADD batch loss to total_loss
        INCREASE batch count
    END FOR

    RETURN average loss
END FUNCTION
```

---

## Function: `validate(...)`

This function evaluates the model on a validation loader.

### What it does

For each batch:

- move data to device,
- run model forward pass,
- compute loss,
- threshold predictions,
- compute Dice and IoU,
- accumulate batch averages,
- return average validation loss, Dice, and IoU.

### Pseudocode

```text
FUNCTION validate(
    model,
    val_loader,
    criterion,
    device,
    threshold = 0.5
)

    SET model to evaluation mode

    SET total_loss = 0
    SET total_dice = 0
    SET total_iou = 0
    SET num_batches = 0

    DISABLE gradients

    FOR each batch of images and masks

        MOVE images to device
        MOVE masks to device

        RUN model forward pass
        COMPUTE loss

        THRESHOLD logits into binary predictions

        COMPUTE Dice
        COMPUTE IoU

        ADD loss to total_loss
        ADD Dice to total_dice
        ADD IoU to total_iou
        INCREASE batch count
    END FOR

    RETURN:
        average validation loss,
        average validation Dice,
        average validation IoU
END FUNCTION
```

---

## Function: `save_checkpoint(...)`

This function saves training checkpoints.

### What it does

- builds a checkpoint dictionary,
- always saves `checkpoint_last.pt`,
- if the checkpoint is the current best, also saves `checkpoint_best.pt`.

### Pseudocode

```text
FUNCTION save_checkpoint(
    model,
    optimizer,
    epoch,
    is_best,
    out_dir,
    metric_value = None
)

    BUILD checkpoint dictionary with:
        epoch
        model state dict
        optimizer state dict

    SAVE checkpoint_last.pt

    IF is_best THEN
        SAVE checkpoint_best.pt

        IF metric_value exists THEN
            PRINT metric value
        END IF
    END IF
END FUNCTION
```

---

## Function: `limit_train_dataset(dataset, limit)`

This helper limits a dataset to its first `N` samples.

### What it does

- if the dataset is already a `Subset`, restrict the indices,
- otherwise create a new `Subset` from the first samples.

### Pseudocode

```text
FUNCTION limit_train_dataset(dataset, limit)

    IF dataset has subset indices THEN
        TAKE first limit indices
        RETURN new subset
    ELSE
        RETURN subset over first min(limit, dataset length) indices
    END IF
END FUNCTION
```

---

## Main Function: `main()`

This is the central function of the file.

It runs the full training workflow.

---

### Step 1: Parse arguments
The script reads all command-line settings.

### Step 2: Set random seeds
It seeds:

- Python random,
- PyTorch,
- NumPy,
- CUDA (if available).

### Step 3: Choose device
It selects CPU or GPU and prints device info.

### Step 4: Prepare output folder
It creates the output directory.

### Step 5: Load dataset
It builds:

- train loader,
- validation loader,
- test loader.

### Step 6: Optional overfit/debug mode
If `--limit-train` is used:

- build a small subset,
- prefer non-empty masks,
- compute subset coverage,
- rebuild train/val/test loaders on that subset.

### Step 7: Build model
Create `UNet2D(in_channels=1, out_channels=1, base_channels=32)`.

### Step 8: Define loss and optimizer
- estimate or use provided `pos_weight`
- create `BCEWithLogitsLoss`
- create Adam optimizer

### Step 9: Configure AMP
If CUDA and `--amp` are enabled, create `GradScaler`.

### Step 10: Training loop
For each epoch:

- train one epoch,
- validate,
- print results,
- update best validation Dice,
- save checkpoints,
- save prediction snapshots.

### Step 11: Final test evaluation
- load best checkpoint,
- evaluate on the test set,
- save test prediction snapshot.

---

## Full Main Function Pseudocode

```text
FUNCTION main()

    PARSE command-line arguments

    SET random seed for:
        Python random
        PyTorch
        NumPy
        CUDA if available

    GET device
    PRINT device info

    IF using CUDA THEN
        ENABLE cuDNN benchmark
    END IF

    CREATE output directory
    PRINT output directory

    PRINT dataset loading message

    CALL make_dataloaders(...)
    GET:
        train_loader
        val_loader
        test_loader

    PRINT number of train, val, and test batches

    IF limit_train is enabled THEN
        PRINT overfit mode message

        CREATE evaluation dataset without augmentations

        SELECT first non-empty masks until limit is reached
        IF not enough non-empty masks THEN
            FALL BACK to first limit samples
        END IF

        COMPUTE mean mask coverage
        PRINT mean coverage

        CREATE limited train dataset with augmentations
        CREATE limited validation dataset without augmentations
        CREATE limited test dataset without augmentations

        REBUILD train_loader
        REBUILD val_loader
        REBUILD test_loader

        PRINT updated batch counts
    END IF

    PRINT model creation message

    CREATE UNet2D model
    MOVE model to device
    PRINT number of model parameters

    IF pos_weight was provided THEN
        USE provided value
        PRINT it
    ELSE
        ESTIMATE positive weight from first training batches
        CLIP estimated value if needed
        PRINT estimated value
    END IF

    CREATE BCEWithLogitsLoss using pos_weight
    CREATE Adam optimizer

    IF CUDA and AMP are enabled THEN
        CREATE GradScaler
        use_amp = True
    ELSE
        scaler = None
        use_amp = False
    END IF

    PRINT training start message

    SET best_val_dice = 0

    FOR epoch from 1 to number of epochs

        CALL train_one_epoch(...)
        STORE train_loss

        CALL validate(...)
        STORE val_loss, val_dice, val_iou

        PRINT epoch summary

        CHECK if this is best validation Dice so far
        UPDATE best value if needed

        IF checkpoint/save condition is met THEN
            SAVE checkpoint
        END IF

        IF snapshot/save condition is met THEN
            CREATE prediction snapshot
        END IF
    END FOR

    PRINT training completed message
    PRINT best validation Dice
    PRINT checkpoint directory

    IF best checkpoint exists THEN
        LOAD best checkpoint
        RESTORE model state

        CALL validate(...) on test_loader
        PRINT test Dice and IoU

        CREATE test prediction snapshot
    END IF

END FUNCTION
```

---

## Notes on Loss and Class Imbalance

The script uses:

```text
BCEWithLogitsLoss
```

with a positive-class weight.

This is important because segmentation masks can be sparse, meaning:

- background pixels are much more common than foreground pixels.

The script can:

- use a provided `--pos-weight`, or
- estimate the weight automatically from the training data.

That helps the model pay more attention to foreground pixels.

---

## Notes on AMP

The script supports automatic mixed precision (AMP) when using CUDA.

This can help:

- reduce memory usage,
- speed up training.

It uses:

- `autocast`
- `GradScaler`

AMP is only enabled if:

- the selected device is CUDA,
- the user passes `--amp`.

---

## Notes on Overfit Mode

The `--limit-train` option is very useful for debugging.

It allows the user to train on a very small subset of the dataset to check whether:

- the model can overfit,
- the loss decreases properly,
- the labels look correct,
- the training pipeline is working.

This is a common debugging strategy in deep learning.

---

## Code Observation

There is one helper in the file:

```text
limit_train_dataset(...)
```

but inside `main()`, the code manually rebuilds the limited datasets instead of calling this helper.

So that helper is valid but not actually used in the current main training path.

That is not a bug, but it is a small design observation.

---

## Relationship to Other Project Files

This file sits at the center of the training stage.

### Relationship summary

- `dataset_generator.py`  
  creates the dataset

- `dataset2d.py`  
  loads the dataset into PyTorch

- `unet2d.py`  
  defines the segmentation model

- `metrics.py`  
  computes Dice and IoU

- `train_2d.py`  
  trains, validates, saves, and tests the model

So this file is the main orchestration script for the 2D segmentation experiments.

---

## Why this file is important in the thesis project

This file is very important because it is the step where the generated synthetic data is actually used to train a segmentation model.

Without this file:

- the project could create data,
- but it could not test whether that data helps train a useful model.

This file is what allows the project to answer key thesis questions about model performance.

In simple terms:

- the earlier files generate the synthetic world,
- `train_2d.py` is where that world is used to train the AI model.
