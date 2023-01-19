import torch
from abc import ABC, abstractmethod


class Model(torch.nn.Module):
    @abstractmethod
    def __init__(self, dim: int, num_classes: int, num_features: int) -> None:
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.dim = dim

    @abstractmethod
    def forward(
        self, x: int, edge_index: int, batch: int, edge_weight=None
    ) -> torch.Tensor:
        pass  # pragma: no cover
