from _typeshed import Incomplete as Incomplete
from carate.evaluation import evaluation as evaluation
from carate.loader.load_data import (
    DatasetObject as DatasetObject,
    StandardDatasetMoleculeNet as StandardDatasetMoleculeNet,
    StandardPytorchGeometricDataset as StandardPytorchGeometricDataset,
)
from typing import Any, Dict, Type, Union

EvaluationMap: Dict[str, evaluation]
EVALUATION_MAP: Incomplete
ModelMap: Dict[str, Any]
MODEL_MAP: Incomplete
DATA_SET_MAP: Dict[
    str,
    Union[
        Type[StandardDatasetMoleculeNet],
        Type[StandardPytorchGeometricDataset],
        Type[StandardPytorchGeometricDataset],
    ],
]

class Config:
    model: Incomplete
    optimizer: Incomplete
    Evaluation: Incomplete
    data_set: Incomplete
    dataset_name: Incomplete
    num_classes: Incomplete
    num_features: Incomplete
    net_dimension: Incomplete
    learning_rate: Incomplete
    dataset_save_path: Incomplete
    test_ratio: Incomplete
    batch_size: Incomplete
    shuffle: Incomplete
    num_cv: Incomplete
    num_epoch: Incomplete
    result_save_dir: Incomplete
    model_save_freq: Incomplete
    def __init__(
        self,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        result_save_dir: str,
        model_save_freq: int,
        Evaluation: ebase.evaluation,
        data_set: DatasetObject,
        model: Any,
        optimizer: str,
        net_dimension: int = ...,
        learning_rate: float = ...,
        dataset_save_path: str = ...,
        test_ratio: int = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
        num_cv: int = ...,
        num_epoch: int = ...,
    ) -> None: ...

class ConfigInitializer:
    @classmethod
    def from_file(cls, file_name: str) -> Config: ...
    @classmethod
    def from_json(cls, json_object: Dict[Any, Any]) -> Config: ...
