import numpy as np

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {
            "time_step": 0,
            "moments_list": [],
        }

    def step(self, gradients, learning_rate=0.001):
        """
        Perform a single Adam optimization step.
        :param gradients: List of Gradient NumPy arrays.
        :param learning_rate: Learning rate.
        :return: List of Step NumPy arrays.
        """
        # Unpack state
        t = self.state["time_step"]
        # Initialize moments at time step 0 to match gradient shape
        if t == 0:
            moments = [{"m": zeros_like_gradient, "v": zeros_like_gradient}
                              for zeros_like_gradient in map(np.zeros_like, gradients)]
        else: # retrieve moments stored in current state
            moments = [{key: np.array(val) for key, val in moments.items()}
                              for moments in self.state["moments_list"]]
        # Increment time step
        t += 1
        # Compute steps
        steps = []
        for layer in range(len(gradients)):
            layer_gradient = gradients[layer]
            # Unpack layer moments
            m, v = moments[layer].values()
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * layer_gradient
            # Update biased second moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (layer_gradient ** 2)
            # Repack moments
            moments[layer].update({"m": m, "v": v})
            # Compute bias-corrected moment estimates
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            # Compute the step array
            step = learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # Add to steps
            steps.append(step)
        # Repack state
        self.state.update({"time_step": t, "moments_list": [{key: val.tolist() for key, val in moment.items()}
                                                            for moment in moments]})
        # Return steps list
        return steps
