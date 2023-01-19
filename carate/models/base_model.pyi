import abc
import torch
from _typeshed import Incomplete as Incomplete
from abc import abstractmethod

from typint import Union

class Model(torch.nn.Module, metaclass=abc.ABCMeta):
    num_classes: Incomplete
    num_features: Incomplete
    dim: Incomplete
    @abstractmethod
    def __init__(self, dim: int, num_classes: int, num_features: int): ...
    @abstractmethod
    def forward(
        self, x, edge_index, batch, edge_weight: Union[Incomplete, None] = ...
    ) -> torch.Tensor: ...
