import os
import pandas as pd

from carate.plotting.plot_classification import (
    plot_classification_algorithm
)

from tests.utils import check_plotting_dir


def test_classification_plot_run(): 

    check_plotting_dir()
    path_to_directory = "./notebooks/data/ENZYMES"
    parameter = "Acc_test"

    plot_classification_algorithm(path_to_directory=path_to_directory, parameter=parameter)
    assert os.path.isdir("./plots")
    assert os.path.isfile(f"./plots/ENZYMES_{parameter}.png")

def test_regression_plot_run(): 
    pass