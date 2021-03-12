import numpy as np
from scipy.special import expit as sigmoid


def head_tail(a):
    return a[0], a[1:]


class layer:
    def __init__(self, dims, l_rate, avg_init, std_init):
        this_dim, next_dims = head_tail(dims)

        if next_dims:
            self.next_layer = layer(next_dims, l_rate, avg_init, std_init)
        else:
            self.next_layer = None

        self.l_rate = l_rate
        self.weights = np.random.normal(
            avg_init, std_init, (this_dim.in_, this_dim.out_)
        )
        self.biases = np.random.normal(avg_init, std_init, (this_dim.out_,))

    def process_layer(self, vect_in):
        out = vect_in @ self.weights
        out += self.biases
        out = sigmoid(out)
        return out

    def train_layer(self, prev_layer_chain_rule, layer_output, layer_input):
        chain_rule = prev_layer_chain_rule * (layer_output @ (1 - layer_output))
        self.weights -= self.l_rate * np.outer(layer_input, chain_rule)
        self.biases -= self.l_rate * chain_rule
        chain_rule = self.weights @ chain_rule
        return chain_rule

    def train(self, vect_in, ans):
        vect_out = self.process_layer(vect_in)
        if self.next_layer:
            prev_chain = self.next_layer.train(vect_out, ans)
        else:
            prev_chain = vect_out - ans
        current_chain = self.train_layer(prev_chain, vect_out, vect_in)
        return current_chain

    def predict(self, vect_in):
        vect_out = self.process_layer(vect_in)
        if self.next_layer:
            return self.next_layer.predict(vect_out)
        else:
            return vect_out
