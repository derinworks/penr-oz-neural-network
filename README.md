# penr-oz-neural-network

A neural network implementation with microservice capabilities built using FastAPI. This repository demonstrates key concepts in neural networks, including forward propagation, backpropagation, and gradient descent, with support for various activation functions, initialization techniques and numerical stability.

## Features
- Layer-wise design for flexible neural network architecture.
- Numerical stability mechanisms
- Customizable training parameters.
  - Multiple weight initialization algorithms: Xavier, He, Gaussian
  - Multiple bias initialization algorithms: Random, Zeros
  - Supports activation functions per layer: Sigmoid, ReLU, Tanh, SoftMax
- Microservice API with endpoints for:
  - Creating models
  - Computing outputs
  - Training models asynchronously with buffering
  - Checking on training progress

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
### Adam Optimizer for efficient convergence
The Adam optimizer is an adaptive learning rate optimization algorithm that combines momentum and RMSProp. 
The updates for weights and biases are computed as follows:

1. Compute the moving averages of gradients and squared gradients:
   ```
   m_t = β1 ⋅ m_(t-1) + (1 - β1) ⋅ g_t
   v_t = β2 ⋅ v_(t-1) + (1 - β2) ⋅ g_t^2
   ```
   where:
   - `g_t` is the gradient at timestep `t`.
   - `m_t` and `v_t` are the first and second moment estimates, respectively.
   - `β1` and `β2` are decay rates for the moments.

2. Correct the bias for the moments:
   ```
   m_t' = m_t / (1 - β1^t)
   v_t' = v_t / (1 - β2^t)
   ```

3. Update weights and biases:
   ```
   Θ_t = Θ_(t-1) - α ⋅ (m_t' / (√v_t' + ε))
   ```
   where `α` is the learning rate and `ε` is a small constant to prevent division by zero.

Adam ensures efficient and stable convergence by dynamically adjusting learning rates for each parameter.

### Numerical Stability in Sigmoid Function
To prevent overflow in the sigmoid function:

```
σ(z) = 1 / (1 + exp(-z))
```

Values of `z` are clipped to the range `[-500, 500]` to avoid numerical instability.

### Softmax Cross Entropy Gradient
When using the softmax function for multi-class classification, the gradient of the cost function with respect to logits `z` is given by:

```
∂J/∂z_i = softmax(z)_i - y_i
```

Where:
- `softmax(z)_i` is the softmax probability for class `i`.
- `y_i` is the true label for class `i` (one-hot encoded).

This gradient is efficient to compute and avoids numerical instability when combined with the log-softmax trick.

## Quickstart Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/derinworks/penr-oz-neural-network.git
   cd penr-oz-neural-network
   ```

2. **Create and Activate a Virtual Environment**:
   - **Create**:
     ```bash
     python -m venv venv
     ```
   - **Activate**:
     - On Unix or macOS:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Service**:
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload
   ```

5. **Interact with the Service**
Test the endpoints using Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Testing and Coverage

To ensure code quality and maintainability, follow these steps to run tests and check code coverage:

1. **Run Tests with Coverage**:
   Execute the following commands to run tests and generate a coverage report:
   ```bash
   coverage run -m pytest
   coverage report
   ```

2. **Generate HTML Coverage Report** (Optional):
   For a detailed coverage report in HTML format:
   ```bash
   coverage html
   ```
   Open the `htmlcov/index.html` file in a web browser to view the report.
