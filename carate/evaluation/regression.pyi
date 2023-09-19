import torch
from _typeshed import Incomplete as Incomplete
from carate.evaluation.base import Evaluation as Evaluation
from carate.loader.load_data import DatasetObject as DatasetObject
from carate.models.base_model import Model as Model
from typing import Any, Tuple, Type

class RegressionEvaluation(Evaluation):
    dataset_name: Incomplete
    dataset_save_path: Incomplete
    test_ratio: Incomplete
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
        result_save_dir: str,
        model_save_freq: int,
    ): ...
    def train(
        self,
        epoch: int,
        model_net: Model,
        device: torch.device,
        train_loader: Type[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        **kwargs: Any,
    ) -> float: ...
    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        epoch: int,
        model_net: Model,
        device: torch.device,
        **kwargs: Any,
    ) -> Tuple[float, float]: ...
