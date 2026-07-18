# Neural Style Transfer using VGG19 and PyTorch

## Overview

This project implements **Neural Style Transfer (NST)** using a pre-trained **VGG19** convolutional neural network in PyTorch. The goal is to generate a new image that preserves the **content** of one image while adopting the **artistic style** of another.

The implementation optimizes the pixels of a target image by minimizing a combination of **content loss** and **style loss**.

---

## Features

- Uses a pre-trained VGG19 model trained on ImageNet.
- Extracts content and style features from different convolution layers.
- Computes style representation using Gram Matrices.
- Optimizes the target image using the Adam optimizer.
- Generates a stylized image after iterative optimization.
- Visualizes the final stylized output using Matplotlib.

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Pillow (PIL)
- Matplotlib

Install the required packages:

```bash
pip install torch torchvision pillow matplotlib
```

---

## Dataset / Input Images

The project requires two images:

- **Content Image** – The image whose structure and objects should be preserved.
- **Style Image** – The image whose artistic style should be transferred.

Example:

```
data/
├── shahid.jpg      # Content Image
└── anime.jpg       # Style Image
```

---

## Project Workflow

```
Content Image
        │
        ▼
   VGG19 Network
        │
        ▼
 Content Features
        │
        │
        ├──────────────┐
        │              │
        ▼              ▼
Target Image      Style Image
     │                 │
     ▼                 ▼
 VGG19 Network    VGG19 Network
     │                 │
     ▼                 ▼
Target Features   Style Features
     │                 │
     ▼                 ▼
Gram Matrix      Gram Matrix
     │                 │
     └──────┬──────────┘
            ▼
 Content Loss + Style Loss
            │
            ▼
      Total Loss
            │
            ▼
 Backpropagation
            │
            ▼
 Update Target Image
```

---

## Image Preprocessing

The images are preprocessed before being passed into the network.

Steps:

- Resize to **224 × 224**
- Convert to tensor
- Add batch dimension

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
```

---

## VGG19 Feature Extractor

The project uses the convolutional layers of a pre-trained **VGG19** network.

```python
models.vgg19(weights=VGG19_Weights.DEFAULT)
```

The classifier layers are not used.

The network is frozen:

```python
for param in vgg.parameters():
    param.requires_grad = False
```

Only the target image is optimized.

---

## Feature Extraction Layers

### Content Layer

```
conv4_2
```

This layer captures the overall structure and objects in the image.

---

### Style Layers

```
conv1_1
conv2_1
conv3_1
conv4_1
conv5_1
```

These layers capture textures, colors, brush strokes, and artistic patterns.

---

## Gram Matrix

The style of an image is represented using a Gram Matrix.

The Gram Matrix measures the correlation between feature maps.

```
Feature Maps
      │
      ▼
Flatten
      │
      ▼
Matrix Multiplication
      │
      ▼
Gram Matrix
```

The Gram Matrix captures texture information while ignoring spatial positions.

---

## Loss Functions

### Content Loss

Content loss measures how much the generated image differs from the content image.

```
Content Loss

=

MSE(
Target Features,
Content Features
)
```

---

### Style Loss

Style loss compares the Gram Matrices of the generated image and the style image.

```
Style Loss

=

MSE(
Gram(Target),
Gram(Style)
)
```

The total style loss is the sum of the losses from all selected style layers.

---

## Total Loss

The optimization objective is

```
Total Loss

=

Content Weight × Content Loss

+

Style Weight × Style Loss
```

Current weights:

| Parameter | Value |
|-----------|------:|
| Content Weight | 1 |
| Style Weight | 100000 |

A larger style weight results in a stronger artistic effect.

---

## Optimization

The target image is initialized as a copy of the content image.

```python
target = content.clone().requires_grad_(True)
```

Only the target image is updated.

Optimizer:

```python
torch.optim.Adam([target], lr=0.001)
```

Training iterations:

```
300 Steps
```

Loss is displayed every 50 iterations.

Example:

```
50 the loss is 4521.32

100 the loss is 2934.81

150 the loss is 1875.46

200 the loss is 1421.37

250 the loss is 1087.91

300 the loss is 921.54
```

---

## Output

The optimized tensor is converted back into an RGB image.

```python
output = target.detach()
```

The final stylized image is displayed using Matplotlib.

---

## Project Structure

```
.
├── data/
│   ├── shahid.jpg
│   └── anime.jpg
├── style_transfer.py
├── README.md
└── requirements.txt
```

---

## Concepts Covered

- Neural Style Transfer
- Transfer Learning
- VGG19
- Feature Extraction
- Content Representation
- Style Representation
- Gram Matrix
- Content Loss
- Style Loss
- Mean Squared Error (MSE)
- Gradient Descent
- Adam Optimizer
- Backpropagation
- PyTorch Tensor Operations
- Image Processing

---

## Applications

This project demonstrates applications of deep learning in image generation, including:

- Artistic image generation
- Photo-to-painting conversion
- Digital artwork creation
- Image stylization
- Creative AI applications

---

## Possible Improvements

- Use higher-resolution images.
- Preserve image colors using color transfer.
- Experiment with different style/content weights.
- Support multiple style images.
- Save the generated image automatically.
- Display intermediate outputs during optimization.
- Use GPU acceleration for faster processing.

---

## Author

Developed using **PyTorch** to demonstrate **Neural Style Transfer (NST)** with a pre-trained **VGG19** model. The project combines the content of one image with the artistic style of another by optimizing a target image using content and style losses.
