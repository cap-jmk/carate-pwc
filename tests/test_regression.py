"""
Perform tests on the functionality of running models against 
a regression dataset 

@author: Julian M. Kleber
"""

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
    #Find out what dataset name does the regression dataset have

    dataset_name = "ZINC_test"
    num_classes = 1
    num_features = 18
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
    num_cv = 2
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
        Evaluation=RegressionEvaluation,
    )
    runner.run(device=device)

