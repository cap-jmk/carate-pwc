"""
Tests about the classficiation abilities of the models 

@author: Julian M. Kleber
"""

import torch

from carate.run import Run
from carate.models.cgc import Net
from carate.load_data import StandardDataLoaderMoleculeNet, StandardDataLoaderTUDataset


def test_classification():
    dataset_name = "ENZYMES"
    num_classes = 6
    num_features = 3
    model = Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = None  # defaults to adams optimizer
    net_dimension = 364
    learning_rate = 0.0005
    dataset_save_path = "data/"
    test_ratio = 10
    batch_size = 64
    shuffle = True
    shrinkage = 51
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
    )
    runner.run(device=device)
