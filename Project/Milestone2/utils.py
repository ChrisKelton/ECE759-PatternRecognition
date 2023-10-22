__all__ = ["probability_error_from_confusion_matrix"]
import numpy as np
from typing import List, Dict, Tuple, Optional


def probability_error_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str],
    *,
    class_probabilities: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise RuntimeError(f"Confusion matrix is not square: '{confusion_matrix.shape}'")

    if len(labels) != confusion_matrix.shape[0]:
        raise RuntimeError(f"Number of labels does not match confusion matrix size. '{len(labels)}' != '{confusion_matrix.shape[0]}'")

    if class_probabilities is None:
        class_probabilities = [1 / confusion_matrix.shape[0]] * confusion_matrix.shape[0]
    if len(class_probabilities) != confusion_matrix.shape[0]:
        raise RuntimeError(f"Number of class probabilities does not match confusion matrix size. '{len(class_probabilities)}' != '{confusion_matrix.shape[0]}'")

    ones_mat_with_zeros_diagonal: np.ndarray = np.ones(confusion_matrix.shape)
    for i in range(confusion_matrix.shape[0]):
        ones_mat_with_zeros_diagonal[i, i] = 0

    incorrect_confusion_matrix = np.multiply(confusion_matrix, ones_mat_with_zeros_diagonal)
    pe_vals: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        pe_vals[label] = np.sum(incorrect_confusion_matrix[idx, :]) / np.sum(confusion_matrix[idx, :])

    total_pe: float = 0
    for pe_val, class_probability in zip(pe_vals.values(), class_probabilities):
        total_pe += class_probability * pe_val

    return total_pe, pe_vals