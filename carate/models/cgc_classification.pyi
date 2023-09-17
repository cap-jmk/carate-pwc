from _typeshed import Incomplete as Incomplete
from carate.models.base_model import Model as Model
from typing import Union

logger: Incomplete

class Net(Model):
    conv1: Incomplete
    conv3: Incomplete
    conv5: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self, dim: int, num_features: int, num_classes: int, heads: int
    ) -> None: ...
    def forward(
        self, x, edge_index, batch, edge_weight: Union[Incomplete, None] = ...
    ): ...
