import numpy as np

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def cross_entropy_loss(predictions, true_labels):
    num_samples = true_labels.shape[0]
    correct_probs = predictions[np.arange(num_samples), true_labels]
    loss = -np.sum(np.log(correct_probs + 1e-8)) / num_samples
    return loss


def initialize_weights():
    W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
    b1 = np.zeros((1, 128))
    W2 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)
    b2 = np.zeros((1, 10))
    return W1, b1, W2, b2


def forward_pass(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache


def backward_pass(cache, W2, true_labels, learning_rate):
    X, Z1, A1, Z2, A2 = cache
    num_samples = X.shape[0]

    dZ2 = A2.copy()
    dZ2[np.arange(num_samples), true_labels] -= 1
    dZ2 /= num_samples

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1_new = X.T @ dZ1
    return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def get_accuracy(predictions, true_labels):
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == true_labels)


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test  = x_test.reshape(-1, 784) / 255.0
    return x_train, y_train, x_test, y_test
