import numpy as np

# from scipy.special import expit as sigmoid


class layer:
    def __init__(self, dims, l_rate):
        this_dim, next_dims = self.head_tail(dims)
        if next_dims:
            self.next_layer = layer(next_dims, l_rate)
        else:
            self.next_layer = None
        self.l_rate = l_rate
        self.weights = self.get_prepped_rand((this_dim.in_, this_dim.out_))
        self.biases = self.get_prepped_rand((this_dim.out_,))

    def head_tail(self, a):
        return a[0], a[1:]

    def get_prepped_rand(self, shape):
        return (np.random.rand(*shape) - 0.5) / 10

    def display(self):
        print(self.biases)
        print(self.weights)
        if self.next_layer:
            self.next_layer.display()

    def process_layer(self, vect_in):
        out = vect_in @ self.weights
        out += self.biases
        out = self.sigmoid(out)
        return out

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

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
