__all__ = ["get_class_mapping", "features_by_class", "split_features_by_class"]
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd


def get_class_mapping(
    df: pd.DataFrame,
    class_name: str,
    prefix_class_strip_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    classes = df[class_name]
    unique_classes, reverse_idx = np.unique(classes, return_inverse=True)
    if prefix_class_strip_name is not None:
        unique_classes = [class_.split(prefix_class_strip_name)[-1] for class_ in unique_classes]
    else:
        unique_classes = list(unique_classes)

    class_mapping: Dict[str, np.ndarray] = {}
    for idx, class_name in enumerate(unique_classes):
        class_mapping[class_name] = np.where(reverse_idx == idx)[0]

    return class_mapping


def features_by_class(series: pd.Series, class_mapping: Dict[str, np.ndarray]) -> Dict[str, List[Any]]:
    series_mapped: Dict[str, List[Any]] = {}
    for class_name, idx in class_mapping.items():
        series_mapped[class_name] = list(series[idx])

    return series_mapped


def split_features_by_class(
    df: pd.DataFrame,
    class_name: str,
    prefix_class_strip_name: Optional[str] = None,
    map_by_feature: bool = True,
) -> Dict[str, Dict[str, List[Any]]]:

    class_mapping = get_class_mapping(df, class_name, prefix_class_strip_name)
    feature_names = list(set(df.columns).difference({class_name}))
    feature_idx: List[int] = []
    for feature_name in feature_names:
        feature_idx.append(list(df.columns).index(feature_name))
    feature_names_arranged: List[str] = []
    for idx in feature_idx:
        feature_names_arranged.append(feature_names[idx])
    feature_names = feature_names_arranged.copy()
    del feature_names_arranged

    feature_mapping: Dict[str, Dict[str, List[Any]]] = {}
    if map_by_feature:
        for feature_name in feature_names:
            feature_mapping[feature_name] = features_by_class(df[feature_name], class_mapping)
    else:
        for feature_name in feature_names:
            mapped_features = features_by_class(df[feature_name], class_mapping)
            for class_name, vals in mapped_features.items():
                feature_mapping.setdefault(class_name, {})[feature_name] = vals

    return feature_mapping
