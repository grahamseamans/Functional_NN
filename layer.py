import numpy as np
from scipy.special import expit as sigmoid


def head_tail(a):
    return a[0], a[1:]


class layer:
    def __init__(self, dims, l_rate, avg_init, std_init):
        """
        Builds the layer and recursively creates layers until we run out of dims
        """
        this, next = head_tail(dims)
        self.l_rate = l_rate
        self.weights = np.random.normal(avg_init, std_init, (this.in_dim, this.out_dim))
        self.biases = np.random.normal(avg_init, std_init, (this.out_dim,))
        if next:
            self.next_layer = layer(next, l_rate, avg_init, std_init)
        else:
            self.next_layer = None

    def process_layer(self, vect_in):
        """
        Matrix multiplication and pariwise addition using the layers weights and biases
        """
        out = vect_in @ self.weights
        out += self.biases
        out = sigmoid(out)
        return out

    def train_layer(self, prev_layer_chain_rule, layer_output, layer_input):
        """
        Modifies weights and biases using the gradient and the learning rate.
        This while thing runs on multivaraible calculus and the chain rule.
        """
        chain_rule = prev_layer_chain_rule * (layer_output @ (1 - layer_output))
        self.weights -= self.l_rate * np.outer(layer_input, chain_rule)
        self.biases -= self.l_rate * chain_rule
        chain_rule = self.weights @ chain_rule
        return chain_rule

    def train(self, vect_in, ans):
        """
        Process each layer using training data, while working it's way down to the bottom.
        It then works it's way back up building the chain rule expression using the
        input's and outputs of the layer with train_layer.
        """
        vect_out = self.process_layer(vect_in)
        if self.next_layer:
            prev_chain = self.next_layer.train(vect_out, ans)
        else:
            prev_chain = vect_out - ans
        current_chain = self.train_layer(prev_chain, vect_out, vect_in)
        return current_chain

    def predict(self, vect_in):
        """
        Works it's way down the net to create a prediction from an input vector.
        """
        vect_out = self.process_layer(vect_in)
        if self.next_layer:
            return self.next_layer.predict(vect_out)
        else:
            return vect_out
