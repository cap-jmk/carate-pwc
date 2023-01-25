import torch
from amarium.utils import make_full_filename as make_full_filename
from carate.models.base_model import Model as Model
from typing import Tuple, Type

def load_model(
    model_path: str, model_net: Type[torch.nn.Module]
) -> Type[torch.nn.Module]: ...
def load_model_training_checkpoint(
    checkpoint_path: str,
    model_net: Type[torch.nn.Module],
    optimizer: Type[torch.optim.Optimizer],
) -> Tuple[Model, torch.optim.Optimizer]: ...
def save_model_training_checkpoint(
    result_save_dir: str,
    dataset_name: str,
    num_cv: int,
    num_epoch: int,
    model_net: Type[torch.nn.Module],
    optimizer: Type[torch.optim.Optimizer],
    loss: float,
) -> None: ...
def save_model(
    result_save_dir: str,
    dataset_name: str,
    num_cv: int,
    num_epoch: int,
    model_net: Type[torch.nn.Module],
) -> None: ...
def load_model_parameters(model_params_file_path: str) -> Dict[Any, Any]: ...
def save_model_parameters(model_net: Type[torch.nn.Module], save_dir: str) -> None: ...
def get_latest_checkpoint(search_dir: str, num_cv: int, epoch: int) -> str: ...
