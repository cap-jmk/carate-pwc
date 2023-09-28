import os
import pandas as pd
import logging

from carate.plotting.base_plots import plot_trajectory_single_algorithm
from carate.plotting.multi_run import plot_all_runs_in_dir

from tests.utils import check_plotting_dir


def test_multi_plot_regression():
    check_plotting_dir()
    path_to_directory = "./notebooks/data/Regression/"
    parameter = "MAE Train"
    save_name = "./plots/multi_run_regression.png"

    plot_all_runs_in_dir(
        base_dir=path_to_directory,
        save_name=save_name,
        val_single=parameter,
        legend_texts=[
            "ALCHEMY_20",
            "ALCHEMY_20_test_training_length_no_norm",
            "ZINC_20",
            "ALCHEMY_20_no_norm",
            "ZINC_20_no_norm",
        ],
    )

    assert os.path.isdir("./plots")
    assert os.path.isfile(save_name)


def test_multi_plot_classification():
    check_plotting_dir()
    path_to_directory = "./notebooks/data/Classification/"
    parameter = "Acc_train"
    save_name = "./plots/multi_run_classification.png"

    plot_all_runs_in_dir(
        path_to_directory,
        save_name=save_name,
        y_lims=(0.009, 0.0009),
        legend_texts=['PROTEINS', 'ENZYMES', 'MCF-7', 'YEAST', 'MOLT-4'],
        val_single=parameter,
    )

    assert os.path.isdir("./plots")
    assert os.path.isfile(save_name)


def test_classification_plot_run():
    check_plotting_dir()
    path_to_directory = "./notebooks/data/Classification/ENZYMES/"
    parameter = "Acc_test"

    plot_trajectory_single_algorithm(
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

    plot_trajectory_single_algorithm(
        path_to_directory=path_to_directory, parameter=parameter, data_name=data_name
    )
    assert os.path.isdir("./plots")
    assert os.path.isfile(
        f"./plots/ALCHEMY_20_test_training_length_no_norm_{parameter}.png"
    )
