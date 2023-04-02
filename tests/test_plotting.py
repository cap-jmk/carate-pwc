import pandas as pd

from carate.plotting.base_plots import (
    plot_range_band_single,
    plot_range_band_multi,
    get_stacked_list,
)


def test_single_plot():
    path_to_directory = "./notebooks/data/ENZYMES"
    result = get_stacked_list(
        path_to_directory=path_to_directory,
        column_name="Acc",
        num_cv=5,
        json_name="ENZYMES.json",
    )
    plot_range_band_single(
        result, key_val="Acc_train", file_name="ENZYMES_accuracy", save_dir="./plots"
    )
