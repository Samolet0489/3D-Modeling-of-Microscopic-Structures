# unet2d.py

## Purpose

This file contains the 2D U-Net model implementation used for image segmentation.

Its purpose is to define the neural network architecture that takes a 2D input image and produces a 2D segmentation output.

The model is a standard U-Net with:

- an encoder path,
- a bottleneck,
- a decoder path,
- skip connections between encoder and decoder levels.

This file is the core model-definition file in the 2D training pipeline.

---

## Main Components

This file contains two main classes:

1. `DoubleConv`
2. `UNet2D`

It also contains a small `__main__` test block at the bottom.

---

## High-Level Role in the Pipeline

At a high level, this file is used like this:

```text
1. Load a 2D microscopy image
2. Pass the image through the U-Net
3. Produce segmentation logits
4. Convert logits into probabilities or binary masks later
5. Compare predictions to target masks during training
```

So this file defines the model that learns the segmentation task.

---

## Inputs

### `DoubleConv(in_channels, out_channels)`

#### Inputs
- `in_channels`
- `out_channels`

#### Output
- A convolutional block that transforms feature maps from `in_channels` to `out_channels`

---

### `UNet2D(in_channels=1, out_channels=1, base_channels=32)`

#### Inputs
- `in_channels`  
  number of input channels

- `out_channels`  
  number of output channels

- `base_channels`  
  number of feature channels in the first encoder block

#### Output
- A full 2D U-Net model

---

## Outputs

### Model forward output

The `forward(...)` method returns:

- a tensor of shape `(B, out_channels, H, W)`

Important:

- the output is **logits**
- no sigmoid activation is applied inside the model

That means thresholding or sigmoid conversion is done later during training or evaluation.

---

## Class: `DoubleConv`

This class defines a standard two-layer convolution block.

### Structure

The block is:

```text
Conv2d -> ReLU -> Conv2d -> ReLU
```

### What it does

- applies one convolution,
- applies ReLU,
- applies a second convolution,
- applies ReLU again.

This is a common building block in U-Net.

### Pseudocode for `DoubleConv.__init__`

```text
FUNCTION DoubleConv.__init__(in_channels, out_channels)

    CALL parent constructor

    CREATE sequential block:
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        ReLU(inplace=True)
        Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        ReLU(inplace=True)

    STORE block as self.double_conv
END FUNCTION
```

### Pseudocode for `DoubleConv.forward`

```text
FUNCTION DoubleConv.forward(x)

    RETURN self.double_conv(x)
END FUNCTION
```

---

## Class: `UNet2D`

This class defines the full 2D U-Net model.

### Main architecture idea

The architecture has three parts:

1. **Encoder**  
   progressively reduces spatial resolution and increases channels

2. **Bottleneck**  
   deepest representation of the image

3. **Decoder**  
   progressively upsamples back to the original resolution

Skip connections connect encoder feature maps to decoder stages at the same scale.

---

## U-Net Architecture Overview

With `base_channels = C`, the model uses:

### Encoder
- `enc1`: `in_channels -> C`
- `enc2`: `C -> 2C`
- `enc3`: `2C -> 4C`
- `enc4`: `4C -> 8C`

Each encoder stage is followed by max pooling.

### Bottleneck
- `8C -> 16C`

### Decoder
- upsample from `16C -> 8C`, concatenate with encoder 4, then decode
- upsample from `8C -> 4C`, concatenate with encoder 3, then decode
- upsample from `4C -> 2C`, concatenate with encoder 2, then decode
- upsample from `2C -> C`, concatenate with encoder 1, then decode

### Final layer
- `C -> out_channels` using a `1x1` convolution

---

## Why skip connections are important

Skip connections pass high-resolution encoder features directly to the decoder.

This helps the model keep fine spatial details, which is especially important for segmentation.

Without skip connections:

- deep features would exist,
- but precise localization would be harder.

So skip connections help combine:

- deep semantic information,
- precise spatial information.

---

## Pseudocode for `UNet2D.__init__`

```text
FUNCTION UNet2D.__init__(
    in_channels = 1,
    out_channels = 1,
    base_channels = 32
)

    CALL parent constructor

    STORE in_channels
    STORE out_channels
    STORE base_channels

    CREATE encoder blocks:
        enc1 = DoubleConv(in_channels, base_channels)
        pool1 = MaxPool2d(2, 2)

        enc2 = DoubleConv(base_channels, base_channels * 2)
        pool2 = MaxPool2d(2, 2)

        enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        pool3 = MaxPool2d(2, 2)

        enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        pool4 = MaxPool2d(2, 2)

    CREATE bottleneck:
        bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

    CREATE decoder upsampling blocks:
        upconv4 = ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        upconv3 = ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        upconv2 = ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        upconv1 = ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        dec1 = DoubleConv(base_channels * 2, base_channels)

    CREATE final output convolution:
        final_conv = Conv2d(base_channels, out_channels, kernel_size=1)

END FUNCTION
```

---

## Pseudocode for `UNet2D.forward`

```text
FUNCTION UNet2D.forward(x)

    # Encoder
    enc1 = enc1 block applied to x
    x = pool1(enc1)

    enc2 = enc2 block applied to x
    x = pool2(enc2)

    enc3 = enc3 block applied to x
    x = pool3(enc3)

    enc4 = enc4 block applied to x
    x = pool4(enc4)

    # Bottleneck
    x = bottleneck(x)

    # Decoder
    x = upconv4(x)
    x = concatenate(x, enc4) along channel dimension
    x = dec4(x)

    x = upconv3(x)
    x = concatenate(x, enc3) along channel dimension
    x = dec3(x)

    x = upconv2(x)
    x = concatenate(x, enc2) along channel dimension
    x = dec2(x)

    x = upconv1(x)
    x = concatenate(x, enc1) along channel dimension
    x = dec1(x)

    # Final output
    x = final_conv(x)

    RETURN x
END FUNCTION
```

---

## Shape Flow Through the Network

Assume the input shape is:

```text
(B, in_channels, H, W)
```

Then the shape changes approximately like this:

```text
Encoder:
(B, C, H, W)
(B, 2C, H/2, W/2)
(B, 4C, H/4, W/4)
(B, 8C, H/8, W/8)

Bottleneck:
(B, 16C, H/16, W/16)

Decoder:
(B, 8C, H/8, W/8)
(B, 4C, H/4, W/4)
(B, 2C, H/2, W/2)
(B, C, H, W)

Output:
(B, out_channels, H, W)
```

This means the network returns an output with the same spatial size as the input.

---

## Final Output Meaning

The final layer uses:

```text
Conv2d(base_channels, out_channels, kernel_size=1)
```

This is a pixel-wise linear mapping from the final feature channels to the output channels.

For binary segmentation:

- `out_channels = 1`
- the output is one logit per pixel

That logit is later converted into:

- probability via sigmoid
- binary prediction via thresholding

---

## `__main__` Test Block

At the bottom of the file, there is a small test routine.

### What it does

1. create the model,
2. create a random input tensor,
3. run a forward pass,
4. print output shape and value range.

This is a quick sanity check that the model runs correctly.

### Pseudocode

```text
IF file is executed as main program THEN

    PRINT test start message

    CREATE UNet2D model
    PRINT model class name

    CREATE random input tensor of shape (2, 1, 128, 128)
    PRINT input shape

    RUN forward pass
    STORE output

    PRINT output shape
    PRINT output dtype
    PRINT output min and max

    PRINT success message

END IF
```

---

## Notes

- The model is a standard 2D U-Net.
- It uses `Conv2d`, `ReLU`, `MaxPool2d`, and `ConvTranspose2d`.
- The convolutions use `kernel_size=3` and `padding=1`, so spatial dimensions stay unchanged inside each block.
- Downsampling happens only through max pooling.
- Upsampling happens through transposed convolutions.
- Output activation is **not** included in the model.
- This is correct when using `BCEWithLogitsLoss`, because that loss expects raw logits.

---

## Relationship to Other Project Files

This model is used directly by the training script.

### Relationship summary

- `dataset2d.py`  
  loads training images and masks

- `unet2d.py`  
  defines the segmentation model

- `metrics.py`  
  evaluates model outputs

- `train_2d.py`  
  trains and validates this model

So `unet2d.py` is the core neural network definition used in the 2D experiments.

---

## Why this file is important in the thesis project

This file is important because it defines the actual AI model that learns from the synthetic data.

The thesis is not only about generating synthetic microscopy data, but also about testing whether such data can improve segmentation performance.

That means a model is needed to learn from the generated data, and this file defines that model.

In simple terms:

- the earlier files create the data,
- `unet2d.py` defines the network that learns from that data.

So this file is the central model-definition file for the 2D segmentation part of the thesis project.
