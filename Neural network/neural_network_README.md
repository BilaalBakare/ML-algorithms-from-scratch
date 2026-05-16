# Neural Network (Multilayer Perceptron)

> Implemented from scratch using NumPy only — tested on load_digits, MNIST, and Fashion-MNIST

---

## Table of Contents
- [Concept](#concept)
- [Architecture](#architecture)
- [Forward Pass](#forward-pass)
- [Activation Functions](#activation-functions)
- [Loss Function](#loss-function)
- [Backpropagation](#backpropagation)
- [Gradient Descent](#gradient-descent)
- [Implementation](#implementation)
- [Training Challenges](#training-challenges)
- [Results](#results)
- [Usage](#usage)
- [Files](#files)
- [Key Takeaways](#key-takeaways)

---

## Concept

A Neural Network is a series of matrix multiplications with non-linearities injected between them. It learns to map input features to output classes by adjusting millions of weights through repeated cycles of prediction, error measurement, and correction.

Unlike KNN which memorizes data or Naive Bayes which counts probabilities — a Neural Network learns internal representations of the data through layers of transformation.

---

## Architecture

The network is defined by a list of layer sizes:

```python
nn([784, 128, 64, 10])
#    ↑    ↑    ↑   ↑
# input  hidden hidden output
```

| Layer | Role | Size determined by |
|---|---|---|
| Input | Holds raw features | Number of features — fixed |
| Hidden | Learns representations | Your choice |
| Output | Final prediction | Number of classes — fixed |

The structure stores **connections between layers** — not the neurons themselves. Each layer transition has a weight matrix and a bias vector:

```
W shape → (n_neurons_in, n_neurons_out)
b shape → (n_neurons_out,)
```

Weights are initialized using **Xavier initialization** to keep activations stable across layers:

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
```

Biases are initialized to zero.

---

## Forward Pass

Data flows through the network layer by layer. At each layer two operations happen:

**Step 1 — Weighted sum:**
```
Z = X @ W + b
```

**Step 2 — Activation function:**
```
A = activation(Z)
```

The output of each layer becomes the input to the next. Both Z and A are stored at every layer — they are needed during backpropagation.

---

## Activation Functions

**ReLU — hidden layers:**
```
f(z) = max(0, z)
```
Introduces non-linearity without being computationally expensive. Any negative value becomes zero — any positive value passes through unchanged.

**Softmax — output layer:**
```
f(zᵢ) = e^(zᵢ - max(z)) / Σ e^(zⱼ - max(z))
```
Converts raw output values into probabilities that sum to 1 — one probability per class. The subtraction of `max(z)` prevents numerical overflow.

---

## Loss Function

**Categorical Cross Entropy** — measures how wrong the predictions are across all classes:

$$L = -\frac{1}{n}\sum_{i}\sum_{j} y_{ij} \cdot \log(\hat{y}_{ij})$$

Since labels are one-hot encoded — only the probability assigned to the correct class contributes to the loss. Confident correct predictions produce tiny loss. Confident wrong predictions produce large loss.

---

## Backpropagation

Backpropagation computes the gradient of the loss with respect to every weight and bias — working backwards through the network using the chain rule.

**The pattern that repeats at every layer:**

```
Step 1 — error signal at output layer:
    dL/dZ = A - y      (softmax + cross entropy simplifies beautifully)

Step 2 — weight gradient:
    dL/dW = previous_activationᵀ @ dL/dZ

Step 3 — bias gradient:
    dL/db = sum(dL/dZ, axis=0)

Step 4 — pass error backwards:
    dL/dA_prev = dL/dZ @ Wᵀ
    dL/dZ_prev = dL/dA_prev * relu_derivative(Z_prev)
```

**The ReLU derivative** is a binary mask — 1 where Z > 0, 0 elsewhere. It blocks the error signal from flowing through dead neurons.

**The intuition:** Each hidden neuron's blame is the sum of all the blame it caused in the next layer — weighted by how strongly it was connected to each neuron there. That is exactly what `dL/dZ @ Wᵀ` computes simultaneously for every neuron.

---

## Gradient Descent

Once gradients are computed — every weight and bias is nudged in the direction that reduces the loss:

```
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

**Mini-batch gradient descent** is used — the dataset is split into batches of 32 samples. Weights update after every batch rather than after the full dataset. This gives more frequent updates and faster convergence on CPU.

**Gradient clipping** caps gradients to a maximum value before applying them:

```python
grad = np.clip(grad, -0.1, 0.1)
```

This prevents exploding gradients from causing catastrophically large weight updates.

---

## Implementation

### Key design decisions

**Matrix based — not node based:** Every neuron in a layer is computed simultaneously via matrix multiplication — not looped over individually. One line replaces thousands of operations.

**Xavier initialization:** Raw random weights cause activations to explode or vanish through deep layers. Xavier scaling keeps the signal stable regardless of network depth.

**Numerical stable softmax:** Subtracting the row maximum before exponentiating prevents overflow when output values are large.

**Log probabilities in loss:** Raw probability multiplication underflows to zero on datasets with many features. Categorical cross entropy uses `log` to keep values numerically stable.

**Gradient clipping:** Prevents the exploding gradient → dying ReLU cascade that kills training in the first epoch.

**Average epoch loss:** Accumulating loss across all batches and dividing by batch count gives an honest picture of training progress — not just the last batch's noise.

---

## Training Challenges

Building this network revealed several important lessons about training neural networks from scratch:

### Exploding Gradients
**Symptom:** Loss oscillates wildly, gradient norms in the millions.
**Cause:** Large input values amplified through layers, raw weight initialization.
**Fix:** Input normalization + Xavier initialization + gradient clipping.

### Dying ReLU
**Symptom:** Gradient norms drop to 0.0 after epoch 1, loss stuck at ~2.302 (random chance for 10 classes).
**Cause:** Exploding gradients in epoch 1 pushed neurons deeply negative — ReLU permanently zeros them out.
**Fix:** Fix exploding gradients first — dying ReLU is a downstream effect.

### Learning Rate Sensitivity
**Too high** → loss oscillates, overshoots minimum.
**Too low** → loss barely moves, needs thousands of epochs.
**Sweet spot for our datasets** → 0.001

### Misleading Loss Reporting
Printing only the last batch loss gave a noisy, misleading signal. Averaging across all batches revealed the true training trend.

---

## Results

### Load Digits (sklearn)
**Dataset:** 1,797 samples, 64 features (8×8 pixels), 10 classes
**Architecture:** `[64, 64, 10]`
**Training:** 200 epochs, lr=0.001, batch_size=32

```
Loss:     0.005 → 0.003   (200 epochs)
Accuracy: 97.11%
```

---

### MNIST
**Dataset:** 70,000 samples, 784 features (28×28 pixels), 10 classes
**Architecture:** `[784, 128, 64, 10]`
**Training:** 10 epochs, lr=0.001, batch_size=32

```
Loss:     0.862 → 0.332   (10 epochs)
Accuracy: 94.54%
```

---

### Fashion-MNIST
**Dataset:** 70,000 samples, 784 features (28×28 pixels), 10 clothing classes
**Architecture:** `[784, 128, 64, 10]`
**Training:** 10 epochs, lr=0.001, batch_size=32

```
Loss:     4.118 → 0.958   (10 epochs)
Accuracy: 82.53%
```

Fashion-MNIST is intentionally harder than digit recognition — clothing categories like shirt, pullover, and coat overlap significantly in pixel space. 82.53% with a simple feedforward network and only 10 epochs is a strong result.

---

## Usage

```python
from NN import nn

# define architecture
model = nn([784, 128, 64, 10])

# train
model.fit(X_train, y_train, epochs=50, learning_rate=0.001, batch_size=32)

# predict
predictions = model.predict(X_test)

# evaluate
accuracy = model.score(predictions, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**Note:** Normalize inputs before training:
```python
X = X / 255.0   # for MNIST and Fashion-MNIST
X = X / 16.0    # for load_digits
```

Labels should be integer arrays — one hot encoding is handled internally by `fit()`.

---

## Files

| File | Purpose |
|---|---|
| `NN.py` | Neural network class — NumPy only |
| `model_Evaluation_And_Comparison.ipynb` | Training and evaluation on load_digits, MNIST, and Fashion-MNIST |
| `README.md` | This file |

---

## Key Takeaways

- A neural network is just matrix multiplications with non-linearities — the complexity comes from training, not the structure
- Backpropagation is blame attribution — each weight inherits blame proportional to how strongly it contributed to the error
- Most training failures trace back to one root cause — fix exploding gradients first and dying ReLU often resolves itself
- Xavier initialization, gradient clipping, and input normalization are not optional extras — they are prerequisites for stable training
- Fashion-MNIST being harder than MNIST reflects a real world truth — visually similar classes are genuinely harder to separate without convolutional layers
- 94.54% on MNIST and 82.53% on Fashion-MNIST with a pure NumPy implementation from scratch — with no deep learning framework — demonstrates the algorithm works correctly
