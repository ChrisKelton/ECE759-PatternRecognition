__all__ = ["get_class_mapping", "features_by_class", "split_features_by_class"]
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd


def get_class_mapping(
    df: pd.DataFrame,
    class_name: str,
    prefix_class_strip_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """

    :param df: dataframe of dataset, with no Index, the first N columns relating to each feature (ea. column with a
        unique feature name) and the last column relating to the class label, named the same as class_name input
    :param class_name: name of column within df containing class label per feature vector
    :param prefix_class_strip_name: optional string to strip from class labels, e.g., 'Iris-' will change 'Iris-setosa'
        to 'setosa'
    :return: indices per class into dataframe
        {
            class0: [0, 1, 2, 3],
            ...,
            classN: [L-5, L-4, L-3, L-2, L-1],
        }, where L = length of dataframe (max = L-1 b/c 0-based indexing)
    """

    classes = df[class_name]
    # get reverse indices into dataframe per unique class
    unique_classes, reverse_idx = np.unique(classes, return_inverse=True)
    # optionally strip class names using prefix_class_strip_name
    if prefix_class_strip_name is not None:
        unique_classes = [class_.split(prefix_class_strip_name)[-1] for class_ in unique_classes]
    else:
        unique_classes = list(unique_classes)

    # generate mapping per class into dataframe
    class_mapping: Dict[str, np.ndarray] = {}
    for idx, class_name in enumerate(unique_classes):
        class_mapping[class_name] = np.where(reverse_idx == idx)[0]

    return class_mapping


def features_by_class(series: pd.Series, class_mapping: Dict[str, np.ndarray]) -> Dict[str, List[Any]]:
    """

    :param series: 1 feature from dataframe holding feature vectors (a single column from a dataframe)
    :param class_mapping: mapping between classes and indices into series
    :return: mapping between class and feature vector values
        {
            class0: class0_feature_vals,
            ...,
            classN: classN_feature_vals,
        }
    """
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
    """

    :param df: dataframe of dataset, with no Index, the first N columns relating to each feature (ea. column with a
        unique feature name) and the last column relating to the class label, named the same as class_name input
    :param class_name: name of column within df containing class label per feature vector
    :param prefix_class_strip_name: optional string to strip from class labels, e.g., 'Iris-' will change 'Iris-setosa'
        to 'setosa'
    :param map_by_feature: boolean to map by feature or map by class.
        If True:
            return {
                      feature0: {
                            class0: class0_feature0_vals,
                            ...,
                            classN: classN_feature0_vals,
                      },
                      ...,
                      featureN: {
                            class0: class0_featureN_vals,
                            ...,
                            classN: classN_featureN_vals,
                      },
                   }
        else:
            return {
                      class0: {
                            feature0: class0_feature0_vals,
                            ...,
                            featureN: class0_featureN_vals,
                      },
                      ...,
                      classN: {
                            featureN: classN_featureN_vals,
                            ...,
                            featureN: classN_featureN_vals,
                      },
                   }
    :return:
    """

    # get class mapping indices into dataframe (figure out which rows in the dataframe belong to what class)
    class_mapping = get_class_mapping(df, class_name, prefix_class_strip_name)

    # get feature names in same ordering as dataframe to preserve feature ordering of input dataframe as using 'set()'
    # may change the ordering of the feature names
    feature_names = list(set(df.columns).difference({class_name}))
    feature_idx: List[int] = []
    for feature_name in feature_names:
        feature_idx.append(list(df.columns).index(feature_name))
    feature_names_arranged: List[str] = []
    for idx in feature_idx:
        feature_names_arranged.append(feature_names[idx])
    feature_names = feature_names_arranged.copy()
    del feature_names_arranged

    # get mapping of features to class or class to features
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
