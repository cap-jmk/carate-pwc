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
    runner.run()  # takes instance attributes as parameters for the run() function
    result_dir_content = os.listdir("tests/results/ZINC_test/data")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 2
    assert (
        r"ZINC_test_0.json" in result_dir_content
        and "ZINC_test_1.json" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ZINC_test/checkpoints")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".pt")]
    assert len(result_dir_content_data) == 4
    assert (
            "ZINC_test_CV-0_Epoch-1.pt" in result_dir_content_data 
        and "ZINC_test_CV-0_Epoch-2.pt" in result_dir_content
        and "ZINC_test_CV-1_Epoch-1.pt" in result_dir_content
        and "ZINC_test_CV-1_Epoch-2.pt" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ZINC_test/model_parameters")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert (
        "model_architecture.json" in result_dir_content_data 
    )
