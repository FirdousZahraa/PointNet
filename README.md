# PointNet

# Introduction
This implementation demonstrates the training and evaluation of a PointNet model for 3D object classification using PyTorch. The model processes 3D point cloud data, extracts features, and performs classification with reasonable accuracy.

## 1. Dataset Handling:
A custom dataset class `PointCloud` is defined to load the data from the `ModelNet10` dataset and provides functionalities for both training and testing sets.

## 2. Model Definition:
The architecture employs a sequence of transformations, including linear layers, batch normalization, and ReLU activations, to extract discriminative features from 3D point cloud data. These features are then used for classification tasks, enabling the model to accurately classify objects based on their spatial structures.

## 3. Training Loop:
The training loop is structured using `PyTorch's DataLoader` to iterate over data batches. A forward pass is executed within the loop to compute predictions, followed by loss calculation, gradient backpropagation, and weight updaation using the Stochastic Gradient Descent (SGD) optimizer.

## 4. Evaluation:
After training, the model's performance is evaluated on the test set using a separate DataLoader. The accuracy of the model on the test data is calculated.

## 5. Memory Management:
It involves moving data and the model to the GPU (cuda) when available and clearing the CUDA cache after evaluation.
