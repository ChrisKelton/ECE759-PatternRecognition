__all__ = ["split_data_by_class"]
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from Project.Milestone1.dataframe import load_iris_dataset
from Project.Milestone1.utils import split_features_by_class


def split_data_by_class(df: pd.DataFrame, N_t: int, class_col: str, rand: bool = False, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param df: dataframe of dataset, with no Index, the first N columns relating to each feature (ea. column with a
        unique feature name) and the last column relating to the class label, named the same as class_name input
    :param N_t: number of training values to retain, remainder will be in test frame
    :param class_col: string referring to name of column in df with class information
    :param rand: get random indices of features
    :param seed: optional integer as seed value for random number generator
    :return: df_train, df_test
    """
    rng = None
    if rand:
        rng = np.random.default_rng(seed)

    if N_t <= 0 and N_t != -1:
        raise RuntimeError(f"Got '{N_t}' training samples to keep. Must be > 0.")

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
    feature_mapping = split_features_by_class(df, class_name=class_col, prefix_class_strip_name="Iris-", map_by_feature=False)

    # create new dataframes for our training & test vectors
    df_train = pd.DataFrame(columns=[*list(list(feature_mapping.values())[0].keys()), class_col])
    df_test = pd.DataFrame(columns=[*list(list(feature_mapping.values())[0].keys()), class_col])
    for class_name, features in feature_mapping.items():
        n_vectors = len(list(features.values())[0])
        # get training indices to extract from total dataset
        if N_t > n_vectors:
            print(f"N_t > number of vectors available in '{class_name}' class. Training set will contain all vectors.")
            train_indices = np.arange(0, n_vectors, dtype=int)
        elif N_t == -1:
            train_indices = np.arange(0, n_vectors, dtype=int)
        elif rng is not None:
            train_indices = set()
            cnt = 0
            while len(train_indices) < N_t and cnt < 100:
                train_indices.update(set(rng.integers(0, n_vectors, size=N_t)))
                cnt += 1
            train_indices = np.asarray(list(train_indices)[:N_t])
        else:
            train_indices = np.arange(0, N_t, dtype=int)
        # get remaining indices into total dataset for testing indices
        test_indices = np.asarray(list(set(np.arange(0, n_vectors, dtype=int)).difference(set(train_indices))))

        # rearrange training & testing features to create numpy array to put into our new training & testing dataframes.
        all_train_features = np.zeros((len(features), len(train_indices)))
        all_test_features = np.zeros((len(features), len(test_indices)))
        for idx, (feature_name, features_) in enumerate(features.items()):
            all_train_features[idx] = np.asarray(features_)[train_indices]
            if len(test_indices) > 0:
                all_test_features[idx] = np.asarray(features_)[test_indices]

        # concatenate our additional feature vectors to our new dataframes
        df_train_new = pd.DataFrame(data=all_train_features.T, columns=list(df_train.columns)[:-1]).assign(
            class_col=[class_name] * len(train_indices))
        df_train_new.rename(columns={"class_col": class_col}, inplace=True)
        df_train = pd.concat([df_train, df_train_new])

        if len(test_indices) > 0:
            df_test_new = pd.DataFrame(data=all_test_features.T, columns=list(df_test.columns)[:-1]).assign(
                class_col=[class_name] * len(test_indices))
            df_test_new.rename(columns={"class_col": class_col}, inplace=True)
            df_test = pd.concat([df_test, df_test_new])

    # fix range problem in dataframe due to concatenating dataframes together
    df_train.index = pd.RangeIndex(0, len(df_train))
    df_test.index = pd.RangeIndex(0, len(df_test))

    return df_train, df_test


def main():
    iris_data_path = Path(
        r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\ProvidedContent\softwareforstudents\iriscorrected.csv")
    cols_to_drop: List[str] = ["Sepal length [cm]", "Sepal width [cm]"]
    iris_df = load_iris_dataset(iris_data_path, cols_to_drop=cols_to_drop)
    df_train, df_test = split_data_by_class(iris_df, -1, "Plant Type")
    print(f"len of train: {len(df_train)}")
    print(f"len of test: {len(df_test)}")


if __name__ == '__main__':
    main()
