"""
Perform tests on the functionality of running models against
a regression dataset

:author: Julian M. Kleber
"""
import os
from typing import Type
import logging

import torch

from amarium.utils import attach_slash, load_json_from_file

from carate.runner.run import RunInitializer
import carate.models.cgc_regression as CGCR
from carate.evaluation.regression import RegressionEvaluation
from carate.loader.load_data import StandardDatasetMoleculeNet, StandardDatasetTUDataset
from tests.utils import check_dir_paths, check_result_files

logging.basicConfig(
    filename="carate.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
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

    verify_len_json(result_dir, dataset_name="ZINC_test")


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

    verify_len_json(result_dir, dataset_name="ZINC_test")


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

    verify_len_json(result_dir, dataset_name="alchemy_full")


def verify_len_json(result_dir: str, dataset_name: str) -> None:
    result_file = attach_slash(result_dir) + f"data/CV_1/{dataset_name}.json"
    result = load_json_from_file(result_file)
    assert len(result["MAE Train"]) == 2
