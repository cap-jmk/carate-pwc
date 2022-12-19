"""
Perform tests on the functionality of running models against 
a regression dataset 

:author: Julian M. Kleber
"""
import os 

import torch

from carate.run import Run
import carate.models.cgc_regression as CGCR
from carate.evaluation.regression import RegressionEvaluation
from carate.load_data import StandardDataLoaderMoleculeNet, StandardDataLoaderTUDataset


import logging

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_regression():

    config_filepath = "tests/config/regression_test_config.py"
    runner = Run.from_file(config_filepath=config_filepath)
    runner.run() # takes instance attributes as parameters for the run() function
    result_dir_content = os.listdir("tests/results/ZINC_test")
    assert len(result_dir_content) == 2
    assert r"ZINC_test_0.csv" in result_dir_content and "ZINC_test_1.csv" in result_dir_content

