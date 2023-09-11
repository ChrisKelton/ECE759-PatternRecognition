from pathlib import Path
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def misclassified_count(test_results: List[Union[float, int]], labels: List[Union[float, int]]) -> Tuple[int, List[int]]:
    error_cnt = 0
    error_indices = []
    if len(test_results) != len(labels):
        raise RuntimeError(f"Got different number of test results to labels.")
    for i in range(len(test_results)):
        if test_results[i] != labels[i]:
            error_cnt += 1
            error_indices.append(i)

    return error_cnt, error_indices


def neuron(w: np.ndarray, x: np.ndarray, act_func: Callable) -> float:
    return act_func(w.T @ x)


def perceptron(
    w_1_0: float,
    w_0_0: float,
    X: List[List[float]],
    p_0: float,
    p_t: Callable,
    stop: int = 1000,
) -> Tuple[np.ndarray, int, List[np.ndarray], List[float], List[int], List[List[int]]]:
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
    training_labels = list(training_labels)
    feature_vector = np.ravel(np.array(X))
    w_1_t = w_1_0
    w_0_t = w_0_0
    w = np.array((w_1_t, w_0_t))

    cost_function_derivative: Callable = lambda x_index: not_training_labels[x_index]*np.array((feature_vector[x_index], 1))
    def cost_function_derivative_sum(error_indices_: List[int]) -> np.ndarray:
        return np.sum([cost_function_derivative(i) for i in error_indices_], axis=0)

    w_t_1: Callable = lambda w_t, p_t_val, cost: w_t - (p_t_val * cost)

    activation_function: Callable = lambda val: np.sign(val)
    def get_results(w: np.ndarray) -> List[float]:
        return [neuron(w, np.array((feature_vector[i], 1)), activation_function) for i in range(len(feature_vector))]

    error_cnt, error_indices = misclassified_count(get_results(w), training_labels)
    error_cnts: List[int] = [error_cnt]
    error_indices_acc: List[List[int]] = [error_indices]
    w_vals: List[np.ndarray] = [w]
    p_vals: List[float] = [p_0]

    iterations = 0
    while error_cnt != 0:
        if iterations == 0:
            w = w_t_1(w, p_0, cost_function_derivative_sum(error_indices))
        elif iterations >= stop:
            break
        else:
            w = w_t_1(w, p_t(iterations), cost_function_derivative_sum(error_indices))
            p_vals.append(p_t(iterations))
        w_vals.append(w)

        error_cnt, error_indices = misclassified_count(get_results(w), training_labels)
        error_cnts.append(error_cnt)
        error_indices_acc.append(error_indices)
        iterations += 1

    return w, iterations, w_vals, p_vals, error_cnts, error_indices_acc


def plot_perceptron_output(
    iterations: int,
    w_vals: List[np.ndarray],
    p_vals: List[float],
    error_cnts: List[int],
    output_path: Path,
):
    w_vals = np.row_stack(w_vals)
    x = np.arange(0, iterations + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].plot(x, error_cnts)
    ax[0].set_ylabel("number errors")
    ax[0].set_xlabel("iterations (t)")

    ax[1].plot(x, w_vals[:, 0], color="r", label="synaptic weights")
    ax[1].plot(x, w_vals[:, 1], color="g", label="threshold")
    ax[1].plot(np.arange(0, iterations), p_vals, color="b", label="step size")
    ax[1].legend(loc="lower right")
    ax[1].set_xlabel("iterations (t)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


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

    w, iterations, w_vals, p_vals, error_cnts, error_indices = perceptron(w_1_0, w_0_0, X, p_0, p_t)
    output_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\HW\HW03\perceptron-output.png")
    plot_perceptron_output(iterations, w_vals, p_vals, error_cnts, output_path)
    a = 0


if __name__ == '__main__':
    main()
