__all__ = ["load_iris_dataset"]
from pathlib import Path
from typing import List

import pandas as pd


def load_iris_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(str(path), index_col=None, header=None)
    column_names: List[str] = ["Petal length [cm]", "Petal width [cm]", "Plant Type"]
    if len(df.columns) == 5:
        column_names = ["Sepal length [cm]", "Sepal width [cm]", *column_names]
    column_names_mapping = {key: val for key, val in zip(list(df.columns), column_names)}
    df.rename(columns=column_names_mapping, inplace=True)
    return df
