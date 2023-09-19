import torch
from _typeshed import Incomplete as Incomplete
from carate.default_interface import DefaultObject as DefaultObject
from carate.evaluation.base import Evaluation as Evaluation
from carate.loader.load_data import DatasetObject as DatasetObject
from carate.models.base_model import Model as Model
from typing import Any, Dict

class Run(DefaultObject):
    dataset_name: Incomplete
    device: Incomplete
    num_classes: Incomplete
    num_features: Incomplete
    Evaluation: Incomplete
    model_net: Incomplete
    net_dimension: Incomplete
    learning_rate: Incomplete
    optimizer: Incomplete
    model_save_freq: Incomplete
    dataset_save_path: Incomplete
    test_ratio: Incomplete
    batch_size: Incomplete
    shuffle: Incomplete
    num_cv: Incomplete
    num_epoch: Incomplete
    result_save_dir: Incomplete
    data_set: Incomplete
    def __init__(
        self,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        result_save_dir: str,
        model_save_freq: int,
        data_set: DatasetObject,
        Evaluation: Evaluation,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        net_dimension: int = ...,
        learning_rate: float = ...,
        dataset_save_path: str = ...,
        test_ratio: int = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
        num_cv: int = ...,
        num_epoch: int = ...,
    ) -> None: ...
    def run(self) -> None: ...

class RunInitializer:
    @classmethod
    def from_file(cls, config_filepath: str) -> Run: ...
    @classmethod
    def from_json(cls, json_object: Dict[Any, Any]) -> Run: ...
