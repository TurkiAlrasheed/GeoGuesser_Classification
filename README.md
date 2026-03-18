# GeoGuesser Classification

This project uses deep learning to classify street-view images into one of six cities.

## Overview

The model was built to learn visual patterns tied to location and predict which city an image came from. It was trained on a dataset of about 9,000 images and fine-tuned from a pretrained EfficientNet-B0 model.

## Model

- EfficientNet-B0 pretrained on ImageNet
- Final classification layer replaced for 6 output classes
- Early feature layers frozen during fine-tuning

## Training

- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Scheduler: cosine annealing
- Loss function: cross-entropy loss
- Batch size: 32
- Epochs: 15

## Data Processing

All images were resized to 224x224 before training.

Augmentations used during training:
- random horizontal flip
- random rotation
- color jitter
- random affine translation

Validation images were only resized and normalized.

## Results

The final model reached:

- 92.7% validation accuracy
- 95.5% training accuracy

It also performed better than the human baseline from the project report.

## Files
- train_vit.py
- predict_vit.py
- README.md

## Running the project

Train the model:

```bash
python train_vit.py
```

Run prediction:

```bash
python predict_vit.py --image path/to/image.jpg
```
