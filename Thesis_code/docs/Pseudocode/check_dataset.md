# check_dataset.py

## Purpose

This file is used for a quick visual sanity check of a generated dataset.

It loads a small random set of image and mask pairs from a dataset folder and displays them in a grid. The image is shown in grayscale, and the corresponding mask is overlaid as a red contour. This helps verify that:

- the dataset folder exists,
- the expected `images/` and `masks/` subfolders are present,
- PNG files are available,
- masks match the images correctly,
- the generated data looks reasonable before training.

---

## Inputs

### Function inputs

#### `data_dir`
Path to the dataset root folder.

Expected structure:

```text
data_dir/
├─ images/
└─ masks/
```

Default value:

```text
./data/synthetic_2d
```

#### `n_samples`
Number of random image/mask pairs to display.

Default value:

```text
6
```

---

## Outputs

The file produces:

- console messages for success or error cases,
- a plotted grid of selected samples,
- a boolean return value:
  - `True` if dataset visualization succeeded,
  - `False` if required files or folders were missing.

When run as a script, it also returns:

- exit code `0` for success,
- exit code `1` for failure.

---

## Main Logic

### High-level description

The script first checks whether the dataset folder and required subfolders exist.  
Then it collects all PNG images from the `images/` folder.  
A small random subset is selected.  
For each selected image, the corresponding mask with the same filename is loaded from the `masks/` folder.  
The image is normalized for display, and the mask is converted into a binary form so its contour can be drawn in red on top of the image.  
Finally, all selected samples are shown in a matplotlib grid.

---

## Pseudocode

```text
FUNCTION check_dataset(data_dir = "./data/synthetic_2d", n_samples = 6)

    CONVERT data_dir into Path object
    SET images_dir = data_dir / "images"
    SET masks_dir = data_dir / "masks"

    IF data_dir does not exist THEN
        PRINT error message
        RETURN False
    END IF

    IF images_dir does not exist THEN
        PRINT error message
        RETURN False
    END IF

    IF masks_dir does not exist THEN
        PRINT error message
        RETURN False
    END IF

    GET all PNG files from images_dir
    SORT image files

    IF no image files are found THEN
        PRINT error message
        RETURN False
    END IF

    PRINT success message with number of found images

    RANDOMLY SELECT up to n_samples files without replacement
    SORT selected indices
    BUILD selected_files list

    CALCULATE n_cols as minimum of 3 and number of selected files
    CALCULATE n_rows as enough rows to display all selected files

    CREATE subplot grid with n_rows and n_cols

    RESHAPE subplot axes if needed so indexing is always consistent

    FLATTEN subplot array for easier iteration

    FOR each selected image file

        LOAD image from images folder
        FIND corresponding mask file with same filename in masks folder
        LOAD mask

        NORMALIZE image to range [0, 1]

        DISPLAY image in grayscale

        CONVERT mask to binary:
            pixels > 128 become 1
            other pixels become 0

        IF mask contains foreground pixels THEN
            DRAW red contour on top of image
        END IF

        SET subplot title to filename
        TURN axis off

    END FOR

    FOR all unused subplot positions
        TURN axis off
    END FOR

    APPLY tight layout
    SHOW figure

    RETURN True
END FUNCTION
```

---

## Command-Line Behavior

When the file is run directly:

1. It creates an argument parser.
2. It accepts:
   - `--data-dir`
   - `--n`
3. It calls `check_dataset(...)` using those values.
4. It exits with:
   - `0` if successful,
   - `1` if unsuccessful.

---

## Command-Line Pseudocode

```text
IF file is executed as main program THEN

    CREATE argument parser

    ADD argument "--data-dir"
        default = "./data/synthetic_2d"

    ADD argument "--n"
        type = integer
        default = 6

    PARSE command-line arguments

    CALL check_dataset(data_dir = args.data_dir, n_samples = args.n)

    IF returned value is True THEN
        EXIT with code 0
    ELSE
        EXIT with code 1
    END IF

END IF
```

---

## Notes

- The script assumes that each image has a mask with exactly the same filename.
- The visualization is meant for manual inspection, not for automated evaluation.
- The mask is not shown as a filled overlay. Only its contour is drawn.
- The threshold `mask > 128` is used to turn the mask into a binary image for contour drawing.
- This file is mainly a dataset debugging and validation utility.

---

## Why this file is useful in the project

This file is important because it gives a quick manual check of dataset quality before model training.  
It helps detect common issues early, for example:

- missing folders,
- empty datasets,
- wrong file naming,
- mismatched image-mask pairs,
- obviously bad masks,
- visually unrealistic generated samples.

This makes it a simple but useful validation step in the synthetic-data pipeline.
