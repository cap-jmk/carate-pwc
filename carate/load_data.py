"""
File for data loading from the standard datasets implemented in the pytorch_geometric #
library. The DataSet loader is implemented as a base class and other subclasses include loaders for standardized benchmarks
as well as custom datasets.

:author: Julian M. Kleber
"""
from typing import Type, Optional, List
from abc import ABC, abstractclassmethod, abstractmethod

import torch
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet, TUDataset

import rdkit as rdkit

from carate.default_interface import DefaultObject

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class DatasetObject(ABC, DefaultObject, torch.utils.data.Dataset):
    """
    Interface for DataLoading objects
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractclassmethod
    def load_data(
        self,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError  # pragme: no cover


class StandardPytorchGeometricDataset(DatasetObject):

    DataSet: torch.utils.data.Dataset

    @classmethod
    def load_data(
        cls,
        dataset_name: str,
        test_ratio: int,
        dataset_save_path: str,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> List[torch.utils.data.DataLoader | torch.utils.data.Dataset]:
        """
        The load_dataset function loads a standard dataset, splits it into a training and testing set,
        and returns the appropriate dataloaders for each. The test_ratio parameter specifies what percentage of
        the original dataset should be used as the testing set. The batch_size parameter specifies how many samples
        should be in each batch.

        :param path:str: Used to Define the path where the dataset is located.
        :param dataset_name:str: Used to Specify which dataset to load.
        :param test_ratio:int: Used to divide the dataset into a training and test set.
        :param batch_size:int: Used to set the batch size for training.
        :return: A train_loader and a test_loader.

        :doc-author: Julian M. Kleber
        """

        if shuffle:
            dataset = cls.DataSet(
                dataset_save_path, name=dataset_name).shuffle()
        else:
            dataset = cls.DataSet(dataset_save_path, name=dataset_name)

        test_dataset = dataset[: len(dataset) // test_ratio]
        train_dataset = dataset[len(dataset) // test_ratio:]
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        return train_loader, test_loader, dataset, train_dataset, test_dataset


class StandardDatasetMoleculeNet(StandardPytorchGeometricDataset):
    """
    Implementation of the Dataset interaface with focus on the models implemented in pytorch_geometric
    and provided by the MoleculeNet collection of datasets.
    """

    DataSet = MoleculeNet

    def __init__(
        self,
        dataset_save_path: str,
        dataset_name: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        The __init__ function is called the constructor and is automatically called when you create a new instance of this class.
        The __init__ function allows us to set attributes that are specific to each object created from the class.
        In our case, we want each data_set object to have a path, dataset_name, test_ratio and batch size attribute.

        :param self: Used to Reference the object to which the function is applied.
        :param path:str: Used to Specify the path to the dataset.
        :param dataset_name:str: Used to Store the name of the data set.
        :param test_ratio:int: Used to Split the data set into a training and testing set.
        :param batch_size:int: Used to Set the batch size.
        :return: The object of the class.

        :doc-author: Julian M. Kleber
        """

        self.dataset_save_path = dataset_save_path
        self.dataset_name = dataset_name
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __repr__(self):
        return "StandardMoleculeNet"


class StandardDatasetTUDataset(StandardPytorchGeometricDataset):
    """
    class for loading standard datasates from the TU Dataset collection implemented
    by PyTorch Geometric.

    author: Julian M. Kleber
    """

    DataSet = TUDataset

    def __init__(
        self,
        dataset_save_path: str,
        dataset_name: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        The __init__ function is called the constructor and is automatically called when you create a new instance of this class.
        The __init__ function allows us to set attributes that are specific to each object created from the class.
        In our case, we want each data_set object to have a path, dataset_name, test_ratio and batch size attribute.

        :param self: Used to Reference the object to which the function is applied.
        :param path:str: Used to Specify the path to the dataset.
        :param dataset_name:str: Used to Store the name of the data set.
        :param test_ratio:int: Used to Split the data set into a training and testing set.
        :param batch_size:int: Used to Set the batch size.
        :return: The object of the class.

        :doc-author: Julian M. Kleber
        """

        self.dataset_save_path = dataset_save_path
        self.dataset_name = dataset_name
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __repr__(self):
        return "StandardTUDataset"
