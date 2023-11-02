__all__ = ["kthperclass_classifier"]
from collections import OrderedDict
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from Project.Milestone1.utils import split_features_by_class


def kthperclass_classifier(
    df: pd.DataFrame,
    test_vectors: np.ndarray,
    k: int,
    class_col: str,
    *,
    return_class_names: bool = True,
) -> List[Union[str, int]]:
    """

    :param df: dataframe of dataset, with no Index, the first N columns relating to each feature (ea. column with a
        unique feature name) and the last column relating to the class label, named the same as class_name input
    :param test_vectors: row vector array with test vector(s)
    :param k: integer value to determine figure-of-merit (FOM) per class
    :param class_col: column name in df corresponding to classes of each feature vector
    :return: labels per test vector(s)
    """
    # split features by class, results in
    # {
    #   class0: {
    #       feature0: class0_feature0_vals,
    #       ...,
    #       featureN: class0_featureN_vals,
    #   },
    #   ...,
    #   classN: {
    #       feature0: classN_feature0_vals,
    #       ...,
    #       featureN: classN_featureN_vals,
    #   },
    # }
    feature_mapping = split_features_by_class(df, class_name=class_col, map_by_feature=False)
    # sort features by name
    feature_mapping = OrderedDict(sorted(feature_mapping.items()))
    # determine number of test vectors
    n_of_test_vectors = test_vectors.shape[0] if test_vectors.ndim > 1 else 1
    # container to hold all figure-of-merit (FOM) values per test vector and training value per class
    test_vectors_distances: np.ndarray = np.zeros((len(feature_mapping), n_of_test_vectors))
    for class_idx, (class_name, features) in enumerate(feature_mapping.items()):
        n_vectors = len(list(features.values())[0])

        # get training data in correct form for calculating distance between test vectors
        features_arr = np.zeros((n_vectors, len(features)))
        for idx, feature in enumerate(features.values()):
            features_arr[:, idx] = feature

        # calculate figure-of-merit (FOM) using cityblock distance
        distances = cdist(test_vectors, features_arr, metric="cityblock")
        # sort distances from lowest to highest per row, i.e., ea. row corresponds to a test vector and ea. row has
        # number of training data per class entries
        distances.sort(-1)
        # retain distance at the kth - 1 entry (corresponds to kth entry due to 0-based indexing) for each test vector
        test_vectors_distances[class_idx] = distances[:, k - 1]

    class_names = list(feature_mapping.keys())
    labels: List[Union[str, int]] = []
    for idx in range(n_of_test_vectors):
        # get minimum cityblock distance for each test vector to each kth closest training value
        # if there's a tie, the 2nd [0] index chooses the smallest class value
        class_idx_min = np.where(test_vectors_distances[:, idx] == np.min(test_vectors_distances[:, idx]))[0][0]
        # either return the original class name or index for class
        if return_class_names:
            labels.append(class_names[class_idx_min])
        else:
            labels.append(class_idx_min)

    return labels


def main():
    class1_vectors = np.asarray([[3, 3], [4, 3], [5, 3]])
    class2_vectors = np.asarray([[1, 2], [1, 4], [1, 3]])
    test_vector = np.asarray([[2, 3]])
    df = pd.DataFrame(data=np.row_stack([class1_vectors, class2_vectors]), columns=["feature0", "feature1"]).assign(class_name=[*[1] * len(class1_vectors), *[2] * len(class2_vectors)])
    df.rename(columns={"class_name": "class"}, inplace=True)
    labels_per_k: Dict[int, List[int]] = {}
    ks: List[int] = [1, 2, 3]
    for k in ks:
        labels = kthperclass_classifier(df, np.row_stack([test_vector, test_vector]), k=k, class_col="class")
        print(f"k={k}:\n"
              f"\tlabels: {labels}\n")
        labels_per_k[k] = labels.copy()


if __name__ == '__main__':
    main()
