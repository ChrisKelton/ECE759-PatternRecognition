__all__ = ["load_iris_dataset"]
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_iris_dataset(
    path: Path,
    cols_to_drop: Optional[List[str]] = None,
    prefix_class_name_strip: Optional[str] = None,
) -> pd.DataFrame:
    """

    :param path: path to .csv file to create pandas dataframe
    :param cols_to_drop: optional list of columns to drop from dataframe
    :param prefix_class_name_strip: optional string to strip prefix for each class name; e.g., 'Iris-'
    :return: loaded iris dataframe
    """
    df = pd.read_csv(str(path), index_col=None, header=None)
    column_names: List[str] = ["Petal length [cm]", "Petal width [cm]", "Plant Type"]
    if len(df.columns) == 5:
        column_names = ["Sepal length [cm]", "Sepal width [cm]", *column_names]
    column_names_mapping = {key: val for key, val in zip(list(df.columns), column_names)}
    df.rename(columns=column_names_mapping, inplace=True)

    if cols_to_drop is not None:
        cols_to_drop_temp: List[str] = []
        for col in cols_to_drop:
            if col in list(df.columns):
                cols_to_drop_temp.append(col)
            else:
                print(f"Column '{col}' not in iris dataframe. Cannot drop...")
        df.drop(cols_to_drop, axis=1, inplace=True)

    if prefix_class_name_strip is not None:
        class_names = df[column_names[-1]]
        class_names = [name.split(prefix_class_name_strip)[-1] for name in class_names]
        df[column_names[-1]] = class_names

    return df
