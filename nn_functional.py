import random
from layer import layer
from scipy import stats
from data_types import datum, dim

train_size = 100000
test_size = 1000
dims = [dim(2, 10), dim(10, 100), dim(100, 10), dim(10, 1)]


def division_examples(n):
    for _ in range(n):
        b = random.randint(1, 2000)
        a = random.randint(0, b)
        yield datum(a / b, (a, b))


def get_error(net, data):
    return net.predict(data.vect_in)[0] - data.ans


if __name__ == "__main__":
    # make net
    net = layer(dims, 0.01, 0, 1)

    # train
    for data in division_examples(train_size):
        net.train(data.vect_in, data.ans)

    # test
    testing_data = [get_error(net, data) for data in division_examples(test_size)]
    print("Predicting a / b, where a <= b")
    print("Error = pred - ans")
    print(stats.describe(testing_data))
