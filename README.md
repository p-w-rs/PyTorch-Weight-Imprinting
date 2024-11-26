# PyTorch Weight Imprinting

This repository implements weight imprinting for deep neural networks in PyTorch, allowing for quick adaptation to new classes with minimal training data.

## Overview

Weight imprinting is a technique that enables neural networks to learn new classes from very few examples by directly setting the weights of the final classification layer based on the network's feature embeddings. This implementation uses a modified MobileNetV3 architecture with L2 normalization and scaled cosine similarity for better few-shot learning performance.

## Repository Structure

- `ImprintingMachine.py`: Main class for performing weight imprinting and inference
- `finetune.py`: Script to prepare a base model for weight imprinting
- `test.py`: Example usage and evaluation on CIFAR-10 dataset

## Preparing a Base Model

The `finetune.py` script shows how to modify a standard model (MobileNetV3 in this case) for weight imprinting:
1. Adds L2 normalization layer
2. Replaces final layer with scaled cosine similarity
3. Trains the model on base classes
4. Periodically performs weight imprinting during training

## Using ImprintingMachine

```python
from ImprintingMachine import ImprintingMachine

# Initialize with pretrained model
machine = ImprintingMachine(
    model_path="path/to/model.pt",    # Pretrained model path
    label_path="path/to/labels.txt",  # List of class labels
    relabel_path="path/to/relabel_dir",                # Path for new class images
    batch_size=128                    # Batch size for imprinting
)

# Run inference on single image
results = machine.run_inference("path/to/image.jpg", top_k=1)

# Add new classes through imprinting
machine.relabel_path = "path/to/new_classes"
dataset = machine.run_imprint()

# Fine-tune on imprinted classes
machine.fine_tune_model(dataset, num_epochs=10, batch_size=32, lr=0.001)

# Save updated model
machine.save_model_pt("path/to/save/model.pt")
machine.save_model_onnx("path/to/save/model.onnx")
```

## Directory Structure for New Classes

For adding new classes, organize images in the following structure:
```
relabel_dir/
    class1/
        image1.jpg
        image2.jpg
    class2/
        image1.jpg
        image2.jpg
```

## Performance

The repository includes evaluation code using CIFAR-10 dataset to demonstrate:
- Base model accuracy
- Imprinting performance on new classes
- Effect of fine-tuning after imprinting
- Impact of reinforcement samples on accuracy

## Requirements

See the `pyproject.toml` file for a list of required packages.
