import torch
from carate.models.base_model import Model as Model

def get_optimizer(
    optimizer_str: str, model_net: Model, learning_rate: float
) -> torch.optim.Optimizer: ...
