"""
Test to verify if the initialization of the config file is correct.

:author: Julian M. Kleber
"""

from carate.load_data import StandardDatasetTUDataset

from carate.config import Config, ConfigInitializer
from carate.models import cgc_classification
from carate.evaluation import classification
from typing import Type


def test_config():
    config_filepath = "tests/config/classification_test_config.py"
    config = ConfigInitializer.from_file(file_name=config_filepath)
    assert config.dataset_name == "ENZYMES"
    assert config.num_classes == 6
    assert config.num_features == 3
    assert config.model == cgc_classification
    assert config.Evaluation.__repr__() == "Classification Evaluation Object"
    assert config.optimizer == "adams"
    assert config.net_dimension == 364
    assert config.learning_rate == 0.0005
    assert config.dataset_save_path == "tests/data/"
    assert config.test_ratio == 10
    assert config.batch_size == 64
    assert config.shuffle == True
    assert config.gamma == 51
    assert config.num_epoch == 2
    assert config.num_cv == 2
    assert config.result_save_dir == "tests/results/ENZYMES"
    assert config.data_set.__repr__() == "StandardTUDataset"
