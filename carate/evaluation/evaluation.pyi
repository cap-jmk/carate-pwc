import torch
from _typeshed import Incomplete
from amarium.utils import check_make_dir as check_make_dir
from carate.default_interface import DefaultObject as DefaultObject
from carate.load_data import (
    DatasetObject as DatasetObject,
    StandardDatasetMoleculeNet as StandardDatasetMoleculeNet,
    StandardPytorchGeometricDataset as StandardPytorchGeometricDataset,
)
from carate.models.base_model import Model as Model
from carate.utils.model_files import (
    get_latest_checkpoint as get_latest_checkpoint,
    load_model_parameters as load_model_parameters,
    load_model_training_checkpoint as load_model_training_checkpoint,
    save_model_parameters as save_model_parameters,
    save_model_training_checkpoint as save_model_training_checkpoint,
)
from typing import Any, Type

class Evaluation(DefaultObject):
    name: str
    dataset_name: Incomplete
    dataset_save_path: Incomplete
    test_ratio: Incomplete
    gamma: Incomplete
    num_epoch: Incomplete
    model_net: Incomplete
    optimizer: Incomplete
    num_classes: Incomplete
    num_cv: Incomplete
    out_dir: Incomplete
    data_set: Incomplete
    batch_size: Incomplete
    shuffle: Incomplete
    device: Incomplete
    result_save_dir: Incomplete
    model_save_freq: Incomplete
    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        result_save_dir: str,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        data_set: DatasetObject,
        test_ratio: int,
        num_epoch: int = ...,
        num_cv: int = ...,
        num_classes: int = ...,
        out_dir: str = ...,
        gamma: float = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
        model_save_freq: int = ...,
    ) -> None: ...
    def cv(
        self,
        num_cv: int,
        num_epoch: int,
        num_classes: int,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        data_set: DatasetObject,
        shuffle: bool,
        batch_size: int,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: int,
        result_save_dir: str,
        model_save_freq: int,
    ): ...
    def train(
        self,
        epoch: int,
        model_net: Model,
        device: torch.device,
        train_loader: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        gamma: int,
    ): ...
    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        epoch: int,
        model_net: Model,
        device: torch.device,
        **kwargs: Any
    ) -> Any: ...
    def save_result(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        data: dict,
    ) -> None: ...
    def save_whole_checkpoint(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        model_net: Type[torch.nn.Module],
        data: dict,
        optimizer: Type[torch.optim.Optimizer],
        loss: float,
    ) -> None: ...
    def save_model_checkpoint(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        model_net: Type[torch.nn.Module],
        optimizer: Type[torch.optim.Optimizer],
        loss: float,
    ) -> None: ...
    def load_model_checkpoint(
        self, checkpoint_path: str, model_net: Model, optimizer=...
    ) -> Model: ...
