"""
Plotting function for the CARATE paper designed for classification algorithms

:author: Julian M. Kleber
"""
from carate.plotting.base_plots import (
    plot_range_band_single,
    plot_range_band_multi,
    get_stacked_list,
)


def plot_classification_algorithm(path_to_directory: str) -> None:
    """
    The plot_classification_algorithm function takes in a path to a directory containing the 
    results ofa classification algorithm and plots the accuracy of that algorithm on both training 
    and testing data.
    
    :param path_to_directory:str: Used to Specify the directory where the results are stored.
    :return: None.
    
    :doc-author: Julian M. Kleber
    """
    
    result = get_stacked_list(
        path_to_directory=path_to_directory,
        column_name="Acc",
        num_cv=5,
        json_name="MOLT-4.json",
    )
    
    legend_text = path_to_directory.split("/")[-1]

    plot_range_band_single(
        result,
        key_val="Acc_train",
        file_name="ENZYMES_accuracy",
        save_dir="./plots",
        alpha=0.4,
        legend_text=legend_text,
    )
