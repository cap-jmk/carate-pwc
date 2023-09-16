import os
import pandas as pd

from carate.plotting.plot_classification import plot_classification_algorithm

from tests.utils import check_plotting_dir


def test_classification_plot_run():
    check_plotting_dir()
    path_to_directory = "./notebooks/data/Classification/ENZYMES/"
    parameter = "Acc_test"

    plot_classification_algorithm(
        path_to_directory=path_to_directory, parameter=parameter
    )
    assert os.path.isdir("./plots")
    assert os.path.isfile(f"./plots/ENZYMES_{parameter}.png")


def test_regression_plot_run():
    check_plotting_dir()
    path_to_directory = (
        "./notebooks/data/Regression/ALCHEMY_20_test_training_length_no_norm/"
    )
    parameter = "MAE Train"
    data_name = "alchemy_full.json"

    plot_classification_algorithm(
        path_to_directory=path_to_directory, parameter=parameter, data_name=data_name
    )
    assert os.path.isdir("./plots")
    assert os.path.isfile(
        f"./plots/ALCHEMY_20_test_training_length_no_norm_{parameter}.png"
    )
