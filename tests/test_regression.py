"""
Perform tests on the functionality of running models against 
a regression dataset 

@author: Julian M. Kleber
"""

import torch

from carate.run import Run
import carate.models.cgc_regression as CGCR
from carate.evaluation.classification import ClassificationEvaluation
from carate.load_data import StandardDataLoaderMoleculeNet, StandardDataLoaderTUDataset


import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def test_regression():
    #Find out what dataset name does the regression dataset have

    dataset_name = "alchemy_full"
    num_classes = 12
    num_features = 6
    model = CGCR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = None  # defaults to adams optimizer
    net_dimension = 364
    learning_rate = 0.0005
    dataset_save_path = "data/"
    test_ratio = 20
    batch_size = 64
    shuffle = True
    shrinkage = 51
    num_epoch = 2
    num_cv = 3
    result_save_dir = "results/"
    runner = Run(
        dataset_name=dataset_name,
        num_features=num_features,
        num_classes=num_classes,
        model=model,
        device=device,
        optimizer=optimizer,
        net_dimension=net_dimension,
        learning_rate=learning_rate,
        dataset_save_path=dataset_save_path,
        DataLoader=StandardDataLoaderTUDataset,
        test_ratio=test_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
        shrinkage=shrinkage,
        num_epoch=num_epoch,
        n_cv=num_cv,
        result_save_dir=result_save_dir,
        Evaluation=ClassificationEvaluation,
    )
    runner.run(device=device)

