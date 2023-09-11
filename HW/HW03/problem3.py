import numpy as np
from typing import Callable, List, Tuple


def misclassified_count(w: np.ndarray, X: np.ndarray, labels: np.ndarray) -> Tuple[int, List[int]]:
    vector_size: int = len(w) - 1
    error_cnt = 0
    error_indices = []
    if vector_size == 1:
        for i in range(len(X) // vector_size):
            x = np.array([X[i], 1]).T
            if np.sign(labels[i]) != np.sign(w @ x):
                error_cnt += 1
                error_indices.append(i)
    else:
        # for i in range(int(np.ceil(len(X) / vector_size))):
        raise NotImplementedError(f"Vector sizes other than 1 not supported at this time.")

    return error_cnt, error_indices

def perceptron(w_1_0: float, w_0_0: float, X: List[List[float]], p_0: float, p_t: Callable, stop: int = 1000) -> Tuple[np.ndarray, int, List[np.ndarray], List[float]]:
    """

    :param w_1_0: initial guess for synaptic weights (only supports size-1 weights)
    :param w_0_0: initial guess for threshold (bias)
    :param X: training vectors of 2 classes
    :param p_0: initial step size
    :param p_t: function of step size per iteration
    :param stop: number of iterations to perform
    :return: final w_1_t & w_0_t to get optimal value to linearly separate both classes in X.
    """
    training_labels = np.concatenate((np.ones(len(X[0])), np.ones(len(X[1])) * -1))
    not_training_labels = training_labels * -1
    X_flat = np.ravel(np.array(X))
    w_1_t = w_1_0
    w_0_t = w_0_0
    w = np.array((w_1_t, w_0_t))

    cost_function: Callable = lambda x_index: not_training_labels[x_index]*X_flat[x_index]
    w_t_1: Callable = lambda w_t, p_t_val, cost: w_t - ((p_t_val * cost) * np.sign(cost))

    iterations = 0
    error_cnt, error_indices = misclassified_count(w, X_flat, training_labels)
    w_vals: List[np.ndarray] = [w]
    p_vals: List[float] = [p_0]
    while error_cnt != 0:
        if iterations == 0:
            w = w_t_1(w, p_0, np.sum([cost_function(i) for i in error_indices]))
        elif iterations >= stop:
            break
        else:
            w = w_t_1(w, p_t(iterations), np.sum([cost_function(i) for i in error_indices]))
            p_vals.append(p_t(iterations))
            w_vals.append(w)

        error_cnt, error_indices = misclassified_count(w, X_flat, training_labels)
        iterations += 1

    return w, iterations, w_vals, p_vals


def main():
    # w_1 (synaptic weights) initial value
    w_1_0 = 1
    # w_0 (threshold) initial value
    w_0_0 = 0
    # training vector for class 1
    X_1 = [-2, 1]
    # training vector for class 2
    X_2 = [3, 5]
    # training vectors
    X = [X_1, X_2]

    # initial step size
    p_0 = 1
    # consequential step sizes, scaled by t-iteration
    p_t: Callable = lambda t: 1/t

    w, iterations, w_vals, p_vals = perceptron(w_1_0, w_0_0, X, p_0, p_t)
    a = 0


if __name__ == '__main__':
    main()
