"""
Plotting function for the CARATE paper designed for classification algorithms

:author: Julian M. Kleber
"""

from typing import Optional

from amarium.utils import attach_slash

from carate.plotting.base_plots import (
    plot_range_band_single,
    plot_range_band_multi,
    get_stacked_list,
)


def plot_classification_algorithm(
    path_to_directory: str,
    parameter: str,
    save_dir: str = "./plots",
    data_name: Optional[str] = None,
) -> None:
    """
    The plot_classification_algorithm function takes in a path to a directory containing the
    results of a classification algorithm and plots the accuracy of that algorithm on both training
    and testing data.

    :param path_to_directory:str: Used to specify the directory where the results are stored.
    :return: None.

    :doc-author: Julian M. Kleber
    """

    path_to_directory = attach_slash(path_to_directory) + "data/"
    legend_text = path_to_directory.split("/")[-3]

    if data_name is None:
        data_name = f"{legend_text}.json"

    result = get_stacked_list(
        path_to_directory=path_to_directory,
        num_cv=5,
        json_name=data_name,
    )

    plot_range_band_single(
        result,
        file_name=f"{legend_text}_{parameter}",
        save_dir=save_dir,
        key_val=parameter,
        alpha=0.4,
        legend_text=legend_text,
    )
