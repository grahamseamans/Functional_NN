import random
from function import layer
import collections
from scipy import stats
from pprint import pprint

datum = collections.namedtuple("datum", ["ans", "vect_in"])

train_size = 100000
test_size = 1000
dims = [[2, 10], [10, 100], [100, 10], [10, 2]]


def rand_int():
    return random.randint(0, 1000)


def make_data(size):
    a = [random.randint(0, 1000) for _ in range(size)]
    b = [random.randint(1000, 2000) for _ in range(size)]
    return [datum(x / y, (x, y)) for (x, y) in zip(a, b)]


def get_error(net, data):
    return net.predict(data.vect_in)[0] - data.ans


if __name__ == "__main__":
    # make data
    train = make_data(train_size)
    test = make_data(test_size)

    # make net
    net = layer(dims, 0.001, 0, 1)

    # train
    for data in train:
        net.train(data.vect_in, data.ans)

    # test
    testing_data = [get_error(net, data) for data in test]
    print("Predicting a / b, where a <= b")
    print("Error = pred - ans")
    pprint(stats.describe(testing_data))
