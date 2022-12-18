"""
Just check if the initialization of the config file is correct. 
"""

from carate.load_data import StandardDataLoaderTUDataset

from carate.config import Config
from carate.models import cgc_classification
from carate.evaluation import classification
def test_config(): 
    config_filepath = "tests/config/classification_test_config.py"
    config = Config.from_file(file_name=config_filepath)
    assert config.dataset_name == "ENZYMES"
    assert config.num_classes == 6
    assert config.num_features == 3
    assert config.model == cgc_classification
    assert config.Evaluation == classification.ClassificationEvaluation
    assert config.optimizer == "adams"
    assert config.net_dimension == 364
    assert config.learning_rate == 0.0005
    assert config.dataset_save_path == "data/"
    assert config.test_ratio == 10
    assert config.batch_size == 64
    assert config.shuffle == True
    assert config.shrinkage == 51
    assert config.num_epoch == 2
    assert config.num_cv == 3
    assert config.result_save_dir == "results/"
    assert config.DataLoader == StandardDataLoaderTUDataset