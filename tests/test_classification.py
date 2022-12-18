"""
Tests about the classficiation abilities of the models 

@author: Julian M. Kleber
"""

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
    runner.run() # takes instance attributes as parameters for the run() function
