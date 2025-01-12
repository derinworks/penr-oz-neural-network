# penr-oz-neural-network

A neural network implementation with microservice capabilities built using FastAPI. This repository demonstrates key concepts in neural networks, including forward propagation, backpropagation, and gradient descent, with support for various activation functions and weight initialization techniques.

## Features
- Multiple weight initialization algorithms (Xavier, He, Gaussian)
- Supports activation functions: Sigmoid, ReLU, Tanh
- Microservice API with endpoints for:
  - Creating models
  - Computing outputs
  - Training models asynchronously
- Numerical stability mechanisms

## Examples: Calculus in Neural Networks

Below are examples demonstrating how calculus is used in the implementation:

### Forward Propagation
Given an activation vector `a`, weights `W`, and biases `b`:

```
z = W ⋅ a + b
```

```
a' = σ(z)
```

Where `σ(z)` is the activation function (e.g., Sigmoid, ReLU, or Tanh).

### Backpropagation: Gradient Calculation
The cost function `J` is defined as:

```
J = (1/m) ∑[i=1 to m] MSE(y_i, ŷ_i)
```

Where `ŷ_i` is the predicted output.

The gradients are computed as:

1. Gradient with respect to output layer activation:
   ```
   ∂J/∂a^(L) = ŷ - y
   ```

2. Gradient with respect to weights:
   ```
   ∂J/∂W^(L) = (∂J/∂a^(L)) ⋅ (a^(L-1))^T
   ```

3. Gradient with respect to biases:
   ```
   ∂J/∂b^(L) = ∂J/∂a^(L)
   ```

### Numerical Stability in Sigmoid Function
To prevent overflow in the sigmoid function:

```
σ(z) = 1 / (1 + exp(-z))
```

Values of `z` are clipped to the range `[-500, 500]` to avoid numerical instability.

## Quickstart Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/derinworks/penr-oz-neural-network.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the microservice:
   ```bash
   uvicorn main:app --reload
   ```

4. Test the endpoints using Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

