from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from dataframe import load_iris_dataset
from plots import scatter_plot_from_list_of_vals

plot_combos_map: Dict[int, int] = {
    1: 4,
    2: 3,
    3: 2,
    4: 1,
}
plot_combos: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]


def plot_feature_combos(
    df: pd.DataFrame,
    base_out_path: Path,
    idx_combo: Optional[List[Tuple[int, int]]] = None,
    plant_types_col_name: str = "Plant Type",
    prefix_plant_type_strip_name: Optional[str] = None,
):
    base_out_path.mkdir(exist_ok=True, parents=True)
    if idx_combo is None:
        idx_combo = plot_combos.copy()

    plant_types = df[plant_types_col_name]
    unique_plant_types, reverse_idx = np.unique(plant_types, return_inverse=True)
    if prefix_plant_type_strip_name is not None:
        unique_plant_types = [plant_type.split(prefix_plant_type_strip_name)[-1] for plant_type in unique_plant_types]
    else:
        unique_plant_types = list(unique_plant_types)
    plant_mapping: Dict[str, np.ndarray] = {}
    for idx, plant_name in enumerate(unique_plant_types):
        plant_mapping[plant_name] = np.where(reverse_idx == idx)[0]

    def feature_values_by_plant_type(series: pd.Series) -> Dict[str, List[Any]]:
        series_mapped: Dict[str, List[Any]] = {}
        for plant_name, idx in plant_mapping.items():
            series_mapped[plant_name] = list(series[idx])

        return series_mapped

    df_cols: List[str] = list(df.columns)
    markers: List[str] = ["^", "x", "s"]
    for combo in idx_combo:
        col0_name = df_cols[plot_combos_map[combo[0]] - 1]
        col0: pd.Series = df[col0_name]
        feature_values0 = feature_values_by_plant_type(col0)

        col1_name = df_cols[plot_combos_map[combo[1]] - 1]
        col1: pd.Series = df[col1_name]
        feature_values1 = feature_values_by_plant_type(col1)

        feature_values: List[Tuple[List[Any], List[Any]]] = []
        for plant_name in plant_mapping.keys():
            feature_values.append((feature_values1[plant_name], feature_values0[plant_name]))

        name = f"{col0_name} (index {plot_combos_map[combo[0]]})(Y) vs. {col1_name} (index {plot_combos_map[combo[1]]})(X)"
        scatter_plot_from_list_of_vals(
            feature_values,
            out_path=base_out_path / f"{name}.png",
            legend=list(plant_mapping.keys()),
            xlabel=col1_name,
            ylabel=col0_name,
            colors=[None, "b", None],
            edgecolors=["black", None, "r"],
            facecolors=["none", None, "none"],
            title=f"fisheriris: {name}",
            marker_styles=markers,
            plot_background_color="#dddddd",
        )


def main():
    iris_data_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\ProvidedContent\softwareforstudents\iriscorrected.csv")
    df = load_iris_dataset(iris_data_path)
    plot_base_out_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE759\Projects\Software\Milestone1\plots")
    plot_feature_combos(df, plot_base_out_path, prefix_plant_type_strip_name="Iris-")


if __name__ == '__main__':
    main()
