import sys
import numpy as np
import csv
from typing import Tuple, Union
import matplotlib.pyplot as plt


def weights(n_hidden_nodes: int,
            x_dimension: int,
            output_dim: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """initialize the weights randomly"""
    w_alpha = np.random.uniform(-0.1, 0.1, size=(n_hidden_nodes, x_dimension))
    w_beta = np.random.uniform(-0.1, 0.1, size=(output_dim, n_hidden_nodes + 1))

    return w_alpha, w_beta


def relu(y: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
    """returns maximum between 0 and given number"""
    return np.maximum(0, y)


def linear(y: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
    """returns the same function"""
    return y


def mse_loss(y_labels: float,
             y_hat: float) -> float:
    """returns mse loss"""
    return np.square(y_labels - y_hat)


def nnforward(y_label: float,
              x_sample: np.ndarray,
              alpha: np.ndarray,
              beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # input layer to hidden layer
    a = np.dot(alpha, x_sample)
    z_star = relu(a)
    z = np.ones(z_star.size + 1)
    z[1:] = z_star

    # hidden layer to output layer
    b = np.dot(beta, z)
    y_hat = linear(b)

    # mean squared loss
    J = mse_loss(y_label, y_hat)

    return a, z, b, y_hat, J


def relu_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0, 1)


def nnbackward(y_train: float,
               x_train: np.ndarray,
               alpha: np.ndarray,
               beta: np.ndarray,
               a: np.ndarray,
               z: np.ndarray,
               b: np.ndarray,
               y_hat: float,
               J: float) -> Tuple[np.ndarray, np.ndarray]:
    # output layer
    delta_output = y_hat - y_train
    # add bias
    delta_w_output = np.empty(shape=beta.shape)
    delta_w_output[:, 0] = delta_output
    delta_w_output[:, 1:] = np.outer(delta_output, z[1:])

    # hidden layer
    delta_hidden = relu_prime(a) * np.dot(delta_output, beta[:, 1:])
    delta_w_hidden = np.empty(shape=alpha.shape)
    delta_w_hidden[:, 0] = delta_hidden
    delta_w_hidden[:, 1:] = np.outer(delta_hidden, x_train[1:])

    return delta_w_hidden, delta_w_output


def train(y_train: np.ndarray,
          x_train: np.ndarray,
          alpha: np.ndarray,
          beta: np.ndarray,
          num_epoch: int,
          gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_J_train = np.empty(num_epoch)
    J_vector_train = np.empty(x_train.shape[0])

    for epoch in range(num_epoch):
        for i, (sample, label) in enumerate(zip(x_train, y_train)):
            a, z, b, y_hat, J = nnforward(label, sample, alpha, beta)
            g_alpha, g_beta = nnbackward(label, sample, alpha, beta, a, z, b, y_hat, J)

            alpha = alpha - (gamma * g_alpha)
            beta = beta - (gamma * g_beta)

            J_vector_train[i] = J
        mean_J_train[epoch] = J_vector_train.mean()

    return alpha, beta, mean_J_train


def predict(x_samples: np.ndarray,
            alpha: np.ndarray,
            beta: np.ndarray) -> np.ndarray:
    y_hat_samples = np.empty(x_samples.shape[0])

    for i, sample in enumerate(x_samples):
        a = np.dot(alpha, sample)
        z_star = relu(a)
        z = np.ones(z_star.size + 1)
        z[1:] = z_star
        b = np.dot(beta, z)
        y_hat = linear(b)

        y_hat_samples[i] = y_hat

    return y_hat_samples


def read_data(filename: str):
    with open(filename, 'rt') as fp:
        reader = csv.reader(fp, delimiter=',')
        x_samples = []  # matrix for inputs
        labels = []  # vector for outputs
        next(reader)  # skip the header

        for sample in reader:
            # iterate over rows, each row is a sample
            # convert string inputs into integers for processing
            x_samples.append(sample[1:])  # add a row to the matrix
            labels.append(sample[1])  # add a sample output to the vector

    x_samples = np.array(x_samples)
    x_samples[:, 0] = 1
    x_samples = x_samples.astype('int8')
    labels = np.array(labels).astype('float')

    return x_samples, labels


def mean_test_loss(true_labels, predicted_labels):
    # to mean test loss
    return np.sum(np.square(true_labels - predicted_labels)) / true_labels.size


def test_accuracy(true_labels, predicted_labels):
    # to calculate prediction error
    mispredictions = np.count_nonzero(true_labels - predicted_labels)
    classification_error = mispredictions / true_labels.size

    return 1.0 - classification_error


if __name__ == '__main__':
    args = sys.argv[1:]
    train_path = args[0]
    test_path = args[1]
    num_epoch = int(args[2])
    hidden_units = int(args[3])
    learning_rate = float(args[4])

    x_train, labels_train = read_data(train_path)
    x_test, labels_test = read_data(test_path)

    init_alpha, init_beta = weights(hidden_units, x_train.shape[1])

    alpha, beta, mean_J_train = train(x_train=x_train,
                                      y_train=labels_train,
                                      alpha=init_alpha, beta=init_beta,
                                      num_epoch=num_epoch, gamma=learning_rate)

    # y_pred_train = predict(x_train, alpha, beta)
    y_pred_test = predict(x_test, alpha, beta)

    test_loss = mean_test_loss(labels_test, y_pred_test)
    print(f'Mean Test Loss: {test_loss}')

    accuracy = test_accuracy(np.round(labels_test, decimals=2), np.round(y_pred_test, decimals=2))
    print(f'Test Accuracy: {accuracy}')

    # PLOT TRAINING LOSS:
    epoch = np.linspace(1, num_epoch + 1, num_epoch)
    plt.plot(epoch, mean_J_train)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MyCode")
    plt.savefig('myCode_trainingloss.png', dpi=100)
