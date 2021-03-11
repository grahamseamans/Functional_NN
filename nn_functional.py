import random
from function import layer
import collections
from scipy import stats

datum = collections.namedtuple("datum", ["ans", "vect_in"])

train_size = 10000
test_size = 10
dims = [[2, 10], [10, 5], [5, 2]]


def rand_int():
    return random.randint(0, 1000)


def make_data(size):
    a = [random.randint(0, 1000) for _ in range(size)]
    b = [random.randint(1000, 2000) for _ in range(size)]
    return [datum(x / y, (x, y)) for (x, y) in zip(a, b)]


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
    learning_data = []
    for data in test:
        prediction = net.predict(data.vect_in)[0]
        learning_data.append(prediction - data.ans)
        """
        if data.ans == round(prediction):
            print("true", end="\t")
        else:
            print("false", end="\t")
        print("pred: ", prediction, " ans: ", data.ans)
        """
    print(stats.describe(learning_data))
