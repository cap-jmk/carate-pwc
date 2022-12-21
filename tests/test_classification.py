"""
Tests about the classficiation abilities of the models 

:author: Julian M. Kleber
"""
import os

import torch

from carate.run import Run
import carate.models.cgc_classification as CGCC
from carate.evaluation.classification import ClassificationEvaluation
from carate.load_data import StandardDataLoaderMoleculeNet, StandardDataLoaderTUDataset
from carate.config import Config

import logging

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_classification():

    config_filepath = "tests/config/classification_test_config.py"
    runner = Run.from_file(config_filepath=config_filepath)
    runner.run()  # takes instance attributes as parameters for the run() function
    result_dir_content = os.listdir("tests/results/ENZYMES/data")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 2
    assert (
        "ENZYMES_0.json" in result_dir_content_data and "ENZYMES_1.json" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ENZYMES/checkpoints")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".pt")]
    assert len(result_dir_content_data) == 4
    assert (
            "ENZYMES_CV-0_Epoch-1.pt" in result_dir_content_data 
        and "ENZYMES_CV-0_Epoch-2.pt" in result_dir_content
        and "ENZYMES_CV-1_Epoch-1.pt" in result_dir_content
        and "ENZYMES_CV-1_Epoch-2.pt" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ENZYMES/model_parameters")
    result_dir_content_data = [x for x in  result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert (
        "model_architecture.json" in result_dir_content_data 
    )
