from pathlib import Path
from typing import Callable, List, Tuple, Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

negative_binary_act_func: Callable = lambda x: -1 if x <= 0 else 1
cost_func_derivative_of_neg_binary_act_func: Callable = lambda not_label, feature_vector: not_label * feature_vector
simple_p_step_func: Callable = lambda t: 1 / t


class Perceptron:
    w: np.ndarray = np.array((1, 0))
    w_vals: List[np.ndarray]
    iteration: int = 0
    error_cnts: List[int]
    p_step: float = 1
    p_step_vals: List[float]
    p_step_func: Callable = simple_p_step_func
    act_func: Callable = negative_binary_act_func
    cost_func_derivative: Callable = cost_func_derivative_of_neg_binary_act_func
    class_mapping: Dict[int, str]

    def __init__(
        self,
        w: np.ndarray,
        iteration: Optional[int] = None,
        p_step: Optional[float] = None,
        p_step_func: Optional[Callable] = None,
        act_func: Optional[Callable] = None,
        cost_func_derivative: Optional[Callable] = None,
        **kwargs,
    ):
        self.w = w
        self.w_vals = kwargs.get("w_vals", [])
        self.error_cnts = kwargs.get("error_cnts", [])
        self.iteration = iteration if iteration is not None else 0
        self.p_step = p_step if p_step is not None else 1
        self.p_step_vals = kwargs.get("p_step_vals", [])
        self.p_step_func = p_step_func if p_step_func is not None else simple_p_step_func
        self.act_func = act_func if act_func is not None else negative_binary_act_func
        self.cost_func_derivative = cost_func_derivative if cost_func_derivative is not None else cost_func_derivative_of_neg_binary_act_func
        self.class_mapping = kwargs.get("class_mapping", {})

    @classmethod
    def neuron(cls, feature_vectors: np.ndarray) -> np.ndarray:
        # [w_1 w_0] @ [x_1 1].T
        return np.row_stack(
            [
                cls.act_func(cls.w.T @ feature_vector) for feature_vector in
                cls.prep_single_dimensional_feature_vectors(feature_vectors)
            ]
        )

    def pretty_print_classification(self, results: np.ndarray):
        for idx, result in enumerate(results):
            out = self.class_mapping[result]
            print(f"Result {idx}: '{out}'")

    @staticmethod
    def misclassified_count(
        results: Union[np.ndarray, List[Union[float, int]]],
        labels: Union[np.ndarray, List[Union[float, int]]],
    ) -> Tuple[int, List[int]]:
        error_cnt = 0
        error_indices = []
        if len(results) != len(labels):
            raise RuntimeError(f"Got different number of test results to labels.")
        for i in range(len(results)):
            if results[i] != labels[i]:
                error_cnt += 1
                error_indices.append(i)

        return error_cnt, error_indices

    @classmethod
    def cost_function(cls, feature_vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # SUM_{X_k in Y}(cost_function_derivative(x_k), where Y is the space of misclassified feature vectors
        return np.sum(
            [
                cls.cost_func_derivative(label, feature_vector)
                for label, feature_vector in zip(labels, feature_vectors)
            ], axis=0
        )

    @staticmethod
    def prep_single_dimensional_feature_vectors(feature_vectors: np.ndarray) -> np.ndarray:
        # get 1-dimensional feature vectors in form of [x_1 1].T
        return np.row_stack([np.array((feature_vector, 1)) for feature_vector in feature_vectors])

    @classmethod
    def gradient_iteration(cls, missed_feature_vectors: np.ndarray, not_labels: np.ndarray):
        if cls.iteration != 0:
            cls.p_step = cls.p_step_func(cls.iteration)
            cls.p_step_vals.append(cls.p_step)
        # w_t_1 = w_t - p_t * cost
        cls.w = cls.w - (cls.p_step * cls.cost_function(cls.prep_single_dimensional_feature_vectors(missed_feature_vectors), not_labels))
        cls.w_vals.append(cls.w)

    @classmethod
    def train(
        cls,
        X: List[List[float]],
        labels: List[List[int]],
        w_1_0: Optional[float] = None,
        w_0_0: Optional[float] = None,
        p_0: Optional[float] = None,
        p_t: Optional[Callable] = None,
        act_func: Optional[Callable] = None,
        cost_func_derivative: Optional[Callable] = None,
        class_mapping: Optional[Dict[int, str]] = None,
        stop_steps: int = 1000,
    ) -> "Perceptron":
        """

        :param X: 1-dimensional training vectors of 2 classes
        :param labels: labels for each training vector X, list of list where each inner list corresponds to a class
        :param w_1_0: initial guess for synaptic weights (only supports size-1 weights)
        :param w_0_0: initial guess for threshold (bias)
        :param p_0: initial step size
        :param p_t: function of step size per iteration
        :param act_func: activation function
        :param cost_func_derivative: derivative of cost function (will accept the inverse label of the feature vector,
                                        and the corresponding feature vector)
        :param class_mapping: mapping from labels to class names for pretty visualization
        :param stop_steps: number of iterations to perform
        :return:
        """
        labels = np.ravel(labels)
        vals = np.unique(labels)
        if len(vals) > 2:
            raise RuntimeError(f"More than 2 classes detected. Got '{len(vals)}' classes.")

        # get inverse of labels for cost function derivative
        class0_idx = np.where(labels == vals[0])[0]
        class1_idx = np.where(labels == vals[1])[0]
        not_labels = labels.copy()
        not_labels[class0_idx] = labels[class1_idx]
        not_labels[class1_idx] = labels[class0_idx]

        feature_vectors = np.ravel(np.array(X))

        if w_1_0 is not None:
            if w_0_0 is None:
                w_0_0 = 0
            cls.w = np.array((w_1_0, w_0_0))
        cls.w_vals = [cls.w]

        if p_0 is not None:
            cls.p_step = p_0
        cls.p_step_vals = [cls.p_step]

        if p_t is not None:
            cls.p_step_func = p_t

        if act_func is not None:
            cls.act_func = act_func

        if cost_func_derivative is not None:
            cls.cost_func_derivative = cost_func_derivative

        error_cnt, error_indices = cls.misclassified_count(cls.neuron(feature_vectors), labels)
        cls.error_cnts = [error_cnt]

        while error_cnt != 0:
            if cls.iteration >= stop_steps:
                break
            else:
                cls.gradient_iteration(feature_vectors[error_indices], not_labels[error_indices])

            error_cnt, error_indices = cls.misclassified_count(cls.neuron(feature_vectors), labels)
            cls.error_cnts.append(error_cnt)
            cls.iteration += 1

        return cls(
            w=cls.w,
            w_vals=cls.w_vals,
            error_cnts=cls.error_cnts,
            iteration=cls.iteration,
            p_step=cls.p_step,
            p_step_vals=cls.p_step_vals,
            p_step_func=cls.p_step_func,
            act_func=cls.act_func,
            cost_func_derivative=cls.cost_func_derivative,
            class_mapping=class_mapping if not None else {},
        )

    def plot_training(self, output_path: Path):
        if self.iteration <= 0:
            raise RuntimeError(f"Unable to plot training data as the number of iterations is '{self.iteration}'.")
        w_vals = np.row_stack(self.w_vals)
        x = np.arange(0, self.iteration + 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].plot(x, self.error_cnts)
        ax[0].set_ylabel("number errors")
        ax[0].set_xlabel("iterations (t)")

        ax[1].plot(x, w_vals[:, 0], color="r", label="synaptic weights")
        ax[1].plot(x, w_vals[:, 1], color="g", label="threshold")
        ax[1].plot(np.arange(0, self.iteration), self.p_step_vals, color="b", label="step size")
        ax[1].legend(loc="lower right")
        ax[1].set_xlabel("iterations (t)")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close()

    def test(self, X: List[float], print: bool = False) -> np.ndarray:
        feature_vectors = np.ravel(np.array(X))
        results = np.ravel(self.neuron(feature_vectors))
        if print:
            self.pretty_print_classification(results)

        return results


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

    output_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\HW\HW03\perceptron-output.png")

    perceptron = Perceptron.train(
        X=X,
        labels=[list(np.ones(len(X[0]))), list(np.ones(len(X[1])) * -1)],
        w_1_0=w_1_0,
        w_0_0=w_0_0,
        p_0=p_0,
        p_t=p_t,
        class_mapping={1: "class 1", -1: "class 2"},
    )
    perceptron.plot_training(output_path)
    perceptron.test(X=[1.6], print=True)


if __name__ == '__main__':
    main()
