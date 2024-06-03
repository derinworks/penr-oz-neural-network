import numpy as np


# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Takes a NumPy array x and returns an array of the same shape


# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)  # Takes a NumPy array x (already sigmoid activated) and returns an array of the same shape


if __name__ == "__main__":
    # Training dataset (XOR problem)
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

    outputs = np.array([[0],
                        [1],
                        [1],
                        [0]])

    # Seed random numbers for reproducibility
    np.random.seed(42)  # 42 is a commonly used seed value

    # Initialize weights randomly with mean 0
    input_layer_neurons = inputs.shape[1]  # Number of input features (2 for XOR problem)
    hidden_layer_neurons = 2  # Arbitrary choice for hidden layer size
    output_layer_neurons = 1  # One output for binary classification

    # Generate random weights and biases for layers
    hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
    output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
    output_bias = np.random.uniform(size=(1, output_layer_neurons))

    # Learning rate
    lr = 0.1

    predicted_output = []

    num_training_cycles = 100000
    print(f"Training for {num_training_cycles} training cycles...")

    # Training loop
    for epoch in range(num_training_cycles):
        # Forward Propagation
        hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
        predicted_output = sigmoid(output_layer_input)

        # Compute error
        error = outputs - predicted_output

        # Backpropagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    # Print final predicted output
    print(f"Final predicted output: {predicted_output}")