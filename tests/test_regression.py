"""
Perform tests on the functionality of running models against
a regression dataset

:author: Julian M. Kleber
"""
import os
import shutil
import torch

from typing import Type
import logging

from amarium.utils import search_subdirs

from carate.run import RunInitializer
import carate.models.cgc_regression as CGCR
from carate.evaluation.regression import RegressionEvaluation
from carate.load_data import StandardDatasetMoleculeNet, StandardDatasetTUDataset


logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_regression_override():

    check_dir_paths()
    config_filepath = "tests/config/regression_test_config_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    # check result files
    result_files, result_dirs = search_subdirs(dir_name="tests/results/ZINC_test/data")
    reference_dirs = [
        "tests/results/ZINC_test/data/CV_0",
        "tests/results/ZINC_test/data/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ZINC_test/data/CV_0/ZINC_test.json",
        "tests/results/ZINC_test/data/CV_1/ZINC_test.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name="tests/results/ZINC_test/checkpoints"
    )

    reference_dirs = [
        "tests/results/ZINC_test/checkpoints/CV_0",
        "tests/results/ZINC_test/checkpoints/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ZINC_test/checkpoints/CV_0/ZINC_test.tar",
        "tests/results/ZINC_test/checkpoints/CV_1/ZINC_test.tar",
    ]
    for name in result_files:
        assert name in reference_files

    result_dir_content = os.listdir("tests/results/ZINC_test/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data


def test_regression_no_override():

    check_dir_paths()

    config_filepath = "tests/config/regression_test_config_no_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    # check result files
    result_files, result_dirs = search_subdirs(dir_name="tests/results/ZINC_test/data")
    reference_dirs = [
        "tests/results/ZINC_test/data/CV_0",
        "tests/results/ZINC_test/data/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ZINC_test/data/CV_0/ZINC_test.json",
        "tests/results/ZINC_test/data/CV_1/ZINC_test.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name="tests/results/ZINC_test/checkpoints"
    )

    reference_dirs = [
        "tests/results/ZINC_test/checkpoints/CV_0",
        "tests/results/ZINC_test/checkpoints/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 4
    reference_files = [
        "tests/results/ZINC_test/checkpoints/CV_0/ZINC_test_Epoch-2.tar",
        "tests/results/ZINC_test/checkpoints/CV_0/ZINC_test_Epoch-1.tar",
        "tests/results/ZINC_test/checkpoints/CV_1/ZINC_test_Epoch-2.tar",
        "tests/results/ZINC_test/checkpoints/CV_1/ZINC_test_Epoch-1.tar",
    ]
    for name in result_files:
        assert name in reference_files

    result_dir_content = os.listdir("tests/results/ZINC_test/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data


def check_dir_paths():
    if os.path.isdir("tests/data"):
        shutil.rmtree("tests/data")
    if os.path.isdir("tests/results"):
        shutil.rmtree("tests/results")
