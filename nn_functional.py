import random
from layer import layer
from data import data
import numpy as np
from data_types import dim

# dims = [dim(784, 10)]
dims = [dim(784, 50), dim(50, 10)]
# dims = [dim(784, 50), dim(50, 30), dim(30, 10)]


def get_error(net, data):
    num_correct = 0
    total = 0
    for (image, label) in data.get_data(mode="test"):
        total += 1
        pred = net.predict(image)
        if np.argmax(pred) == np.argmax(label):
            num_correct += 1
    return num_correct / total


if __name__ == "__main__":
    epochs = 100
    mnist = data()

    # make net
    net = layer(dims, 0.001)

    # train
    for _ in range(epochs):
        for image, label in mnist.get_data(mode="train"):
            net.train(image, label)
        print(get_error(net, mnist))

    # test
    print(get_error(net, mnist))
