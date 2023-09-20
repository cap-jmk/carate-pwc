"""
Tests about the classficiation abilities of the models

:author: Julian M. Kleber
"""
import os
import shutil

import torch

from amarium.utils import search_subdirs, load_json_from_file, delete_file


import carate.models.cgc_classification as CGCC
from carate.evaluation.classification import ClassificationEvaluation
from carate.loader.load_data import StandardDatasetMoleculeNet, StandardDatasetTUDataset
from carate.config_adapter.config import Config
from carate.runner.run import RunInitializer

import logging

from typing import Type

logging.basicConfig(
    filename="carate.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_classification_override():

    check_dir_paths()
    config_filepath = "tests/config/classification_test_config_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    runner.run()  # takes instance attributes as parameters for the run() function

    # check result files
    result_files, result_dirs = search_subdirs(dir_name="tests/results/ENZYMES/data")
    reference_dirs = [
        "tests/results/ENZYMES/data/CV_0",
        "tests/results/ENZYMES/data/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ENZYMES/data/CV_0/ENZYMES.json",
        "tests/results/ENZYMES/data/CV_1/ENZYMES.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name="tests/results/ENZYMES/checkpoints"
    )

    reference_dirs = [
        "tests/results/ENZYMES/checkpoints/CV_0",
        "tests/results/ENZYMES/checkpoints/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ENZYMES/checkpoints/CV_0/ENZYMES.tar",
        "tests/results/ENZYMES/checkpoints/CV_1/ENZYMES.tar",
    ]
    for name in result_files:
        assert name in reference_files

    result_dir_content = os.listdir("tests/results/ENZYMES/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data


def test_classification_no_override():
    check_dir_paths()
    config_filepath = "tests/config/classification_test_config_no_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    runner.run()  # takes instance attributes as parameters for the run() function

    # check result files
    result_files, result_dirs = search_subdirs(dir_name="tests/results/ENZYMES/data")
    reference_dirs = [
        "tests/results/ENZYMES/data/CV_0",
        "tests/results/ENZYMES/data/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ENZYMES/data/CV_0/ENZYMES.json",
        "tests/results/ENZYMES/data/CV_1/ENZYMES.json",
    ]
    result_cv1 = load_json_from_file("tests/results/ENZYMES/data/CV_1/ENZYMES.json")
    assert len(result_cv1["Loss"]) == 2
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name="tests/results/ENZYMES/checkpoints"
    )

    reference_dirs = [
        "tests/results/ENZYMES/checkpoints/CV_0",
        "tests/results/ENZYMES/checkpoints/CV_1",
    ]

    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 4
    reference_files = [
        "tests/results/ENZYMES/checkpoints/CV_0/ENZYMES_Epoch-1.tar",
        "tests/results/ENZYMES/checkpoints/CV_0/ENZYMES_Epoch-2.tar",
        "tests/results/ENZYMES/checkpoints/CV_1/ENZYMES_Epoch-1.tar",
        "tests/results/ENZYMES/checkpoints/CV_1/ENZYMES_Epoch-2.tar",
    ]
    for name in result_files:
        assert name in reference_files

    result_dir_content = os.listdir("tests/results/ENZYMES/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data


def check_dir_paths():
    if os.path.isdir("tests/data"):
        shutil.rmtree("tests/data")
    if os.path.isdir("tests/results"):
        shutil.rmtree("tests/results")
