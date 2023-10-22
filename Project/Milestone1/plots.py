__all__ = ["scatter_plot_from_list_of_vals"]
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Any, Tuple, Union


def scatter_plot_from_list_of_vals(
    vals: List[Tuple[List[Any], List[Any]]],
    out_path: Path,
    legend: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Optional[Union[List[str], str]] = None,
    edgecolors: Optional[Union[List[str], str]] = None,
    facecolors: Optional[Union[List[str], str]] = None,
    title: Optional[str] = None,
    marker_styles: Optional[Union[str, List[str]]] = None,
    plot_background_color: Optional[str] = None,
):
    if colors is None:
        colors = [None] * len(vals)
    elif isinstance(colors, str):
        colors = [colors] * len(vals)
    if len(colors) != len(vals):
        raise RuntimeError(f"Got different number of colors than input values. '{len(colors)}' != '{len(vals)}'")

    if edgecolors is None:
        edgecolors = [None] * len(vals)
    elif isinstance(edgecolors, str):
        edgecolors = [edgecolors] * len(vals)
    if len(edgecolors) != len(vals):
        raise RuntimeError(f"Got different number of edgecolors than input values. '{len(edgecolors)}' != '{len(vals)}'")

    if facecolors is None:
        facecolors = [None] * len(vals)
    elif isinstance(facecolors, str):
        facecolors = [facecolors] * len(vals)
    if len(facecolors) != len(vals):
        raise RuntimeError(f"Got different number of facecolors than input invalues. '{len(facecolors)}' != '{len(vals)}'")

    if marker_styles is None:
        marker_styles = [None] * len(vals)
    elif isinstance(marker_styles, str):
        marker_styles = [marker_styles] * len(vals)
    if len(marker_styles) != len(vals):
        raise RuntimeError(f"Got different number of marker styles than input values. '{len(marker_styles)}' != '{len(vals)}'")

    if legend is None:
        legend = [None] * len(vals)
    if len(legend) != len(vals):
        raise RuntimeError(f"Got different number of legend labels than input values. '{len(legend)}' != '{len(vals)}'")

    fig = plt.figure(figsize=(10, 10))
    for idx, val in enumerate(vals):
        plt.scatter(val[0], val[1], c=colors[idx], edgecolors=edgecolors[idx], facecolors=facecolors[idx], marker=marker_styles[idx], label=legend[idx])

    if legend is not None:
        plt.legend(legend)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    fig.savefig(str(out_path), facecolor=plot_background_color)
    plt.close()
