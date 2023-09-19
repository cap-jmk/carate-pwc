import torch
from _typeshed import Incomplete as Incomplete
from carate.evaluation.base import Evaluation as Evaluation
from carate.loader.load_data import DatasetObject as DatasetObject

class ClassificationEvaluation(Evaluation):
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
        model_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_set: DatasetObject,
        model_save_freq: int,
        test_ratio: int,
        num_epoch: int = ...,
        num_cv: int = ...,
        num_classes: int = ...,
        out_dir: str = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
    ) -> None: ...
