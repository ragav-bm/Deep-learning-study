# Deep Learning Study: From Scratch to PyTorch

This repository contains deep learning architectures, layers, and optimization techniques implemented from the ground up in pure **NumPy** (without deep learning frameworks), alongside a modular **PyTorch** pipeline.

> **Course Context:**  
> This project was developed during my Master's degree (2nd semester, **November 2024 – January 2025**) as part of an Advanced Deep Learning lab course. All architecture implementations and mathematical derivations were evaluated and defended during oral lab reviews.

---

## Key Highlights

- **Pure NumPy Implementations:** Forward and backward passes (analytical gradients), optimizers, and activations built from mathematical first principles without PyTorch/TensorFlow in the core modules.
- **Architectures Covered:** Multi-Layer Perceptrons (MLP), 1D/2D Convolutional Networks (CNN), and Elman Recurrent Neural Networks (RNN) with Backpropagation Through Time (BPTT).
- **Optimization & Regularization:** Custom implementations of Adam, SGD with Momentum, Batch Normalization, Inverted Dropout, and L1/L2 weight penalties.
- **Production Pipeline in PyTorch:** Custom ResNet model design, modular training harness with validation checkpointing, early stopping, and ONNX export support.

---

## Repository Structure

```text
├── Feed Forward Neural Networks/
│   ├── Base.py                 # Abstract base layer interface
│   ├── FullyConnected.py       # Dense layer with forward & analytical backprop
│   ├── ReLU.py & SoftMax.py    # Activation functions & gradient propagation
│   ├── Loss.py                 # Cross-Entropy loss with numerical stability (eps)
│   ├── Optimizers.py           # Stochastic Gradient Descent (SGD)
│   └── NeuralNetwork.py        # Sequential network container & training loop
│
├── Convolutional Neural Network/
│   ├── Conv.py                 # 1D & 2D Convolution with custom stride & padding
│   ├── Pooling.py              # Max-pooling with spatial index tracking for backprop
│   ├── Flatten.py              # Multidimensional tensor reshaping
│   ├── Initializers.py         # Xavier / Glorot & He (Kaiming) weight initializations
│   └── Optimizers.py           # Adam & SGD with Momentum optimizers
│
├── Regularization and Recurrent/
│   ├── RNN.py                  # Elman RNN with Backpropagation Through Time (BPTT)
│   ├── BatchNormalization.py   # 1D/2D BatchNorm with running stats & phase toggling
│   ├── Dropout.py              # Inverted Dropout with train/eval modes
│   ├── Constraints.py          # L1 and L2 weight regularizers
│   └── Sigmoid.py & TanH.py    # Additional activation layers
│
└── PyTorch/
    ├── model.py                # Custom ResNet & residual blocks (skip connections)
    ├── data.py                 # Custom Dataset loader with preprocessing & transforms
    ├── trainer.py              # Training loop, early stopping, Macro F1, ONNX export
    └── train.py                # Training entry point & hyperparameter setup
```

---

## Module Overview

### 1. Feed-Forward Neural Networks (From Scratch)
- Explicit matrix calculus for fully connected layers ($Y = XW + b$).
- Numerically stable Softmax combined with Categorical Cross-Entropy.
- Modular forward/backward pass chaining.

### 2. Convolutional Neural Networks (From Scratch)
- 1D and 2D spatial correlation/convolution forward passes and gradient backpropagation with respect to inputs, weights, and biases.
- 2D Max Pooling tracking spatial argmax coordinates for gradient routing.
- Parameter initializers: He and Xavier/Glorot variance scaling.
- Optimizers: Adam (with bias corrections $\hat{m}_t, \hat{v}_t$) and SGD with Momentum.

### 3. Regularization & Recurrent Networks (From Scratch)
- **Elman RNN:** Sequence processing unrolled over time with state retention (`memorize` mode) and full BPTT gradient propagation across recurrent weights.
- **Batch Normalization:** Explicit mean and variance normalization across batch and spatial dimensions, learning scale ($\gamma$) and shift ($\beta$) parameters with test-time moving averages.
- **Inverted Dropout & L1/L2 Regularization:** In-training activation masking and weight penalty gradients.

### 4. PyTorch Deep Learning Pipeline
- Built-from-scratch **ResNet** with configurable residual blocks and 1x1 projection shortcuts.
- Reusable `Trainer` class supporting:
  - Mini-batch training and evaluation loops
  - Validation-loss-driven checkpointing and early stopping
  - Macro F1-score evaluation
  - ONNX model export for deployment

---

## Tech Stack

- **Languages & Core Libraries:** Python 3, NumPy, SciPy
- **Deep Learning Framework:** PyTorch, Torchvision
- **Evaluation & Utilities:** Scikit-Learn, Matplotlib, Pandas

