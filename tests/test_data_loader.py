"""Test module to test the Config, Runner, Dataset pattern
The initialization is a bit weird and needs proper testing leaving no 
doubt 

:author: Julian M. Kleber
"""
from carate.load_data import StandardDatasetTUDataset, StandardDatasetMoleculeNet


def test_tu_data_set():

    dataset_save_path = "tests/data/"
    dataset_name = "ZINC_test"
    test_ratio = 10
    batch_size = 64
    shuffle = True

    loader = StandardDatasetTUDataset(
        dataset_save_path=dataset_save_path,
        dataset_name=dataset_name,
        test_ratio=test_ratio,
        batch_size=batch_size,
    )

    train_loader, test_loader, dataset, train_dataset, test_dataset = loader.load_data(
        dataset_name, test_ratio, dataset_save_path
    )

    assert str(loader) == "StandardTUDataset"


def test_molnet_data_set():

    dataset_save_path = "tests/data/"
    dataset_name = "sider"
    test_ratio = 10
    batch_size = 64
    shuffle = True

    loader = StandardDatasetMoleculeNet(
        dataset_save_path=dataset_save_path,
        dataset_name=dataset_name,
        test_ratio=test_ratio,
        batch_size=batch_size,
    )

    train_loader, test_loader, dataset, train_dataset, test_dataset = loader.load_data(
        dataset_name, test_ratio, dataset_save_path
    )

    assert str(loader) == "StandardMoleculeNet"
