import numpy as np


class data:
    def __init__(self):
        self.train_image = self.read_mnist_file("t10k-images-idx3-ubyte")
        self.train_label = self.read_mnist_file("t10k-labels-idx1-ubyte")
        self.test_image = self.read_mnist_file("train-images-idx3-ubyte")
        self.test_label = self.read_mnist_file("train-labels-idx1-ubyte")
        self.prep_input()

    def read_mnist_file(self, path):
        with open(path, "rb") as f:
            _, _, d_type, num_dimensions = np.fromfile(f, dtype=np.dtype(">B"), count=4)
            dims = np.fromfile(f, dtype=np.dtype(">u4"), count=num_dimensions)
            data = np.fromfile(f, dtype=np.dtype(">B"))
            data = np.reshape(data, dims)
            return data

    def prep_input(self):
        self.train_image = self.flatten_normalize_add_bias(self.train_image)
        self.test_image = self.flatten_normalize_add_bias(self.test_image)
        self.train_label = self.one_hot_np_array(self.train_label, 10)
        self.test_label = self.one_hot_np_array(self.test_label, 10)

    def flatten_normalize_add_bias(self, data):
        data = np.reshape(data, (data.shape[0], 28 ** 2))
        data = data / 255
        return data

    # https://stackoverflow.com/a/49790223
    def one_hot_np_array(self, data, num_classes):
        return np.eye(num_classes)[data]

    def get_data(self, mode):
        if mode == "test":
            return zip(self.test_image, self.test_label)
        elif mode == "train":
            return zip(self.train_image, self.train_label)
        else:
            raise ValueError("Argument to data must be either: test or train")
