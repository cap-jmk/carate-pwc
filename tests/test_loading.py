"""
The test aims to evaluate the power of the Data loader


@author = Julian M. Kleber
"""
import sys, os
import pytest
from carate.load_data import StandardDataLoader


@pytest.fixture(scope="session")
def data_set_classification():
    path = "./data/"
    data_set_name = "ENZYMES"
    test_ratio = 10
    batch_size = 64
    DataLoader = StandardDataLoader(path, data_set_name, test_ratio, batch_size)
    train_loader, test_loader = DataLoader.load_datset()
    return train_loader, test_loader


@pytest.fixture(scope="session")
def data_set_classification():
    path = "./data/"
    data_set_name = "alchemy_full"
    test_ratio = 10
    batch_size = 64
    DataLoader = StandardDataLoader(path, data_set_name, test_ratio, batch_size)
    train_loader, test_loader = DataLoader.load_datset()
    return train_loader, test_loader
