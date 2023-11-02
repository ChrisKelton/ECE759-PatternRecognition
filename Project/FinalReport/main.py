from pathlib import Path
from typing import Dict, Union, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from Project.FinalReport.kthperclass_classifier import kthperclass_classifier
from Project.FinalReport.split_data import split_data_by_class
from Project.Milestone2.utils import probability_error_from_confusion_matrix, load_reduced_iris_dataframe


def part_one(iris_data_path: Path, out_path: Path):
    out_path.mkdir(exist_ok=True, parents=True)
    iris_df = load_reduced_iris_dataframe(iris_data_path)
    class_map: Dict[str, int] = {"setosa": 0, "versicolor": 1, "virginica": 2}
    test_cases: Dict[int, Dict[str, Union[np.ndarray, str]]] = {
        1: {"vector": np.asarray([[2.0, 0.8]]), "class": "setosa"},
        2: {"vector": np.asarray([[4.0, 0.8]]), "class": "versicolor"},
        3: {"vector": np.asarray([[6.5, 2.5]]), "class": "virginica"},
        4: {"vector": np.asarray([[4.5, 1.7]]), "class": "virginica"},
        5: {"vector": np.asarray([[4.8, 1.8]]), "class": "virginica"},
        6: {"vector": np.asarray([[5.0, 1.8]]), "class": "versicolor"},
        7: {"vector": np.asarray([[5.0, 1.5]]), "class": "virginica"},
    }
    test_vals = [val for val in [vals["vector"] for vals in test_cases.values()]]
    true_classes = [val for val in [vals["class"] for vals in test_cases.values()]]
    for k in [1, 3]:
        labels = kthperclass_classifier(iris_df, np.row_stack(test_vals), k, class_col="Plant Type")
        for test_num, (label, true_class) in enumerate(zip(labels, true_classes)):
            print(f"k={k}, test case {test_num + 1}:\n"
                  f"\ttrue class = {true_class}\n"
                  f"\tpredicted class = {label}\n")
        confusion_mat = confusion_matrix(true_classes, labels)
        df_k = pd.DataFrame(confusion_mat, index=list(class_map.keys()), columns=list(class_map.keys()))
        print(f"k={k} confusion matrix:\n{df_k}\n")

        confusion_matrix_path = out_path / f"k={k}-confusion_matrix.png"
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_k, annot=True)
        plt.savefig(str(confusion_matrix_path))
        plt.tight_layout()
        plt.close()

        total_pe, pe_vals = probability_error_from_confusion_matrix(confusion_mat, list(class_map.keys()))
        print(f"k={k}:\n"
              f"\ttotal probability error: {total_pe:.3f}")
        for class_name, pe_val in pe_vals.items():
            print(f"\t{class_name} probability error: {pe_val:.3f}")
        print("\n")


def part_two(iris_data_path: Path, out_path: Path, N_t_vals: Optional[List[int]] = None):
    if N_t_vals is None:
        N_t_vals = [30]

    out_path.mkdir(exist_ok=True, parents=True)
    iris_df = load_reduced_iris_dataframe(iris_data_path)

    knn_classifier_preprocess: Callable = lambda x: np.round(x * 10).astype(int)
    max_k: int = 17
    ks = np.arange(1, max_k + 1)

    for N_t in N_t_vals:
        pe_kthper: List[float] = []
        pe_knn: List[float] = []

        df_train, df_test = split_data_by_class(iris_df, N_t, class_col="Plant Type")
        feature_names = list(df_test.columns)[:2]
        train_vectors = np.column_stack([df_train[feature_names[0]], df_train[feature_names[1]]])
        test_vectors = np.column_stack([df_test[feature_names[0]], df_test[feature_names[1]]])
        true_test_class = list(df_test["Plant Type"])
        true_train_class = list(df_train["Plant Type"])

        vals, cnts = np.unique(true_train_class, return_counts=True)
        class_probabilities: List[float] = [*(cnts / sum(cnts))]
        # class_probabilities = None
        for k in ks:
            labels_kthper = kthperclass_classifier(df_train, test_vectors, k, class_col="Plant Type")
            confusion_mat_kthper = confusion_matrix(true_test_class, labels_kthper)
            total_pe_kthper, _ = probability_error_from_confusion_matrix(
                confusion_mat_kthper,
                ["setosa", "versicolor", "virginica"],
                class_probabilities=class_probabilities,
            )
            pe_kthper.append(total_pe_kthper)

            knn = KNeighborsClassifier(n_neighbors=k, metric="cityblock")
            knn.fit(knn_classifier_preprocess(train_vectors), true_train_class)
            labels_knn = knn.predict(knn_classifier_preprocess(test_vectors))
            confusion_mat_knn = confusion_matrix(true_test_class, labels_knn)
            total_pe_knn, _ = probability_error_from_confusion_matrix(
                confusion_mat_knn,
                ["setosa", "versicolor", "virginica"],
                class_probabilities=class_probabilities,
            )
            pe_knn.append(total_pe_knn)

        fig = plt.figure(figsize=(10, 7))
        plt.plot(ks, pe_knn, linestyle="-", marker="o", color="b", label="kNN", markerfacecolor="none")
        plt.plot(ks, pe_kthper, linestyle="-", marker="x", color="orange", label="kthperclass")
        plt.xlabel("k value")
        plt.ylabel("Probability of classification error")
        plt.title(f"irisf34 dataset: train 1:{N_t}, test {N_t + 1}:50")
        plt.ylim([-0.01, min(max([*pe_knn, *pe_kthper]) + 0.01, 1.00)])
        plt.legend()
        plt.tight_layout()
        fig.savefig(str(out_path / f"Nt={N_t}--classification-errors.png"))
        plt.close()

        print(f"average knn probability error: {np.average(pe_knn):.3f}")
        print(f"average kthper probability error: {np.average(pe_kthper):.3f}")


def main():
    base_out_path = Path(
        r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\FinalReport\outputs")
    iris_data_path = Path(
        r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\ProvidedContent\softwareforstudents\iriscorrected.csv")
    part_one(iris_data_path, base_out_path / "part-1")
    part_two(iris_data_path, base_out_path / "part-2", N_t_vals=[30])


if __name__ == '__main__':
    main()
