"""
Perform tests on the functionality of running models against
a regression dataset

:author: Julian M. Kleber
"""
from typing import Type
import logging

import torch

from carate.run import RunInitializer
import carate.models.cgc_regression as CGCR
from carate.evaluation.regression import RegressionEvaluation
from carate.load_data import StandardDatasetMoleculeNet, StandardDatasetTUDataset
from tests.utils import check_dir_paths, check_result_files

logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_regression_multitaksing():
    check_dir_paths()
    config_filepath = "tests/config/regression_alchemy_test.py"
    run_title = "ALCHEMY_test"
    data_set_name = "alchemy_full"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = f"tests/results/{run_title}"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    check_result_files(
        result_dir=result_dir,
        data_set_name=data_set_name,
        run_title=run_title,
        override=True,
    )


def test_regression_override():
    check_dir_paths()
    config_filepath = "tests/config/regression_test_config_override.py"
    run_title = "ZINC_test"
    data_set_name = "ZINC_test"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = f"tests/results/{run_title}"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    check_result_files(
        result_dir=result_dir, run_title=run_title, data_set_name=data_set_name
    )


def test_regression_no_override():
    check_dir_paths()

    config_filepath = "tests/config/regression_test_config_no_override.py"
    run_title = "ZINC_test"
    data_set_name = "ZINC_test"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = f"tests/results/{run_title}/"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    check_result_files(
        result_dir=result_dir,
        run_title=run_title,
        data_set_name=data_set_name,
        override=True,
    )
