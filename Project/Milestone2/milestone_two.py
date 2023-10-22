from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from Project.Milestone1.dataframe import load_iris_dataset
from Project.Milestone1.utils import split_features_by_class
from utils import probability_error_from_confusion_matrix


def knn_prediction(r: float, k: int, N: int) -> float:
    return (1/(np.pi * (r ** 2))) * (k / N)


def milestone_two(
        iris_data_path: Path,
        base_out_path: Optional[Path] = None,
        *,
        allow_floats: bool = False,
):
    df = load_iris_dataset(iris_data_path)
    cols_to_drop: List[str] = ["Sepal length [cm]", "Sepal width [cm]"]
    df.drop(cols_to_drop, axis=1, inplace=True)
    feature_mapping = split_features_by_class(df, "Plant Type", "Iris-", map_by_feature=False)
    features: List[np.ndarray] = []
    y_truth: List[int] = []
    y_truth_map: Dict[int, str] = {}
    for idx, (class_name, feature_map) in enumerate(feature_mapping.items()):
        shape = (len(feature_map[list(feature_map.keys())[0]]), len(feature_map))
        y_truth.extend([idx + 1] * shape[0])
        y_truth_map[idx + 1] = class_name
        features_: np.ndarray = np.zeros(shape)
        for idx_, values in enumerate(feature_map.values()):
            features_[:, idx_] = values
        features.append(features_)
    features: np.ndarray = np.concatenate(features)

    preprocessing: Callable = lambda x: x
    if not allow_floats:
        preprocessing = lambda x: np.round(x * 10).astype(int)

    k_vals: List[int] = [1, 3]
    values_to_predict: List[List[float]] = [[2.0, 0.8], [4.0, 0.8], [6.5, 2.5], [4.5, 1.7], [4.8, 1.8], [5.0, 1.8],
                                            [5.0, 1.5]]
    test_cases: Dict[int, Dict[str, str]] = {}
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(preprocessing(features), y_truth)
        for idx, val_to_predict in enumerate(values_to_predict):
            prediction = knn.predict(preprocessing(np.asarray(val_to_predict).reshape(-1, 2)))
            test_cases.setdefault(idx + 1, {})[f"k={k}"] = y_truth_map[prediction[0]]

    df = pd.DataFrame(
        columns=["Test case", "Petal length", "Petal width", "True Class", "Pred Class k=1", "Pred Class k=3"])
    true_classes = ["Setosa", "Versicolor", "Virginica", "Virginica", "Virginica", "Versicolor", "Virginica"]
    for idx, (test_case, prediction_out_map) in enumerate(test_cases.items()):
        k_vals = list(prediction_out_map.values())
        df.loc[len(df)] = [test_case, *values_to_predict[idx], true_classes[idx], *k_vals]

    df.set_index("Test case", inplace=True)
    print(df)
    if base_out_path is not None:
        df.to_csv(str(base_out_path / "Prediction-Test-Cases.csv"))

    inv_truth_map: Dict[str, int] = {}
    for key, val in y_truth_map.items():
        inv_truth_map[val] = key
    y_true = np.asarray([inv_truth_map[str(key).lower()] for key in df["True Class"]])
    y_true_unique = list(inv_truth_map.keys())

    k_1_confusion_matrix = confusion_matrix(y_true, np.asarray([inv_truth_map[key] for key in df["Pred Class k=1"]]))
    df_k_1 = pd.DataFrame(k_1_confusion_matrix, index=y_true_unique, columns=y_true_unique)

    k_3_confusion_matrix = confusion_matrix(y_true, np.asarray([inv_truth_map[key] for key in df["Pred Class k=3"]]))
    df_k_3 = pd.DataFrame(k_3_confusion_matrix, index=y_true_unique, columns=y_true_unique)

    print(f"k=1 confusion matrix:\n{df_k_1}\n")
    print(f"k=3 confusion matrix:\n{df_k_3}\n")

    if base_out_path is not None:
        k_1_confusion_matrix_path = base_out_path / "k=1-confusion_matrix.png"
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_k_1, annot=True)
        plt.savefig(str(k_1_confusion_matrix_path))
        plt.close()

        k_3_confusion_matrix_path = base_out_path / "k=3-confusion_matrix.png"
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_k_3, annot=True, )
        plt.savefig(str(k_3_confusion_matrix_path))
        plt.close()

    k_1_total_pe, k_1_pe_vals = probability_error_from_confusion_matrix(k_1_confusion_matrix, y_true_unique)
    print(f"k=1:\n"
          f"\ttotal probability error: {k_1_total_pe:.3f}")
    for class_name, pe_val in k_1_pe_vals.items():
        print(f"\t{class_name} probability error: {pe_val:.3f}")

    k_3_total_pe, k_3_pe_vals = probability_error_from_confusion_matrix(k_3_confusion_matrix, y_true_unique)
    print(f"k=3:\n"
          f"\ttotal probability error: {k_3_total_pe:.3f}")
    for class_name, pe_val in k_3_pe_vals.items():
        print(f"\t{class_name} probability error: {pe_val:.3f}")


def main():
    iris_data_path = Path(
        r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\ProvidedContent\softwareforstudents\iriscorrected.csv")
    # allow_floats: List[bool] = [True, False]
    allow_floats: List[bool] = [False]
    for allow_floats_ in allow_floats:
        print(f"{'Allow floats' if allow_floats_ else 'Dont allow floats'}")
        out_results = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\Milestone2\allow-floats=" + str(allow_floats_))
        out_results.mkdir(exist_ok=True, parents=True)
        milestone_two(iris_data_path, base_out_path=out_results, allow_floats=allow_floats_)


if __name__ == '__main__':
    main()
