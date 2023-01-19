import abc
import torch
from _typeshed import Incomplete as Incomplete
from abc import ABC
from carate.default_interface import DefaultObject as DefaultObject
from typing import List, Union

logger: Incomplete

class DatasetObject(
    ABC, DefaultObject, Type[torch.utils.data.Dataset], metaclass=abc.ABCMeta
):
    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool,
    ) -> None: ...
    def load_data(
        self,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool,
    ) -> None: ...

class StandardPytorchGeometricDataset(DatasetObject, metaclass=abc.ABCMeta):
    DataSet: torch.utils.data.Dataset
    @classmethod
    def load_data(
        cls,
        dataset_name: str,
        test_ratio: int,
        dataset_save_path: str,
        batch_size: int = ...,
        shuffle: bool = ...,
    ) -> List[Union[torch.utils.data.DataLoader, torch.utils.data.Dataset]]: ...

class StandardDatasetMoleculeNet(StandardPytorchGeometricDataset):
    DataSet: Incomplete
    dataset_save_path: Incomplete
    dataset_name: Incomplete
    test_ratio: Incomplete
    batch_size: Incomplete
    shuffle: Incomplete
    def __init__(
        self,
        dataset_save_path: str,
        dataset_name: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool = ...,
    ) -> None: ...

class StandardDatasetTUDataset(StandardPytorchGeometricDataset):
    DataSet: Incomplete
    dataset_save_path: Incomplete
    dataset_name: Incomplete
    test_ratio: Incomplete
    batch_size: Incomplete
    shuffle: Incomplete
    def __init__(
        self,
        dataset_save_path: str,
        dataset_name: str,
        test_ratio: int,
        batch_size: int,
        shuffle: bool = ...,
    ) -> None: ...
