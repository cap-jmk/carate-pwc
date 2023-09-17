import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool,  TransformerConv

import numpy as np
import sklearn.metrics as metrics

import logging


from carate.models.base_model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class Net(Model):
    """
    The Net is the core algorithm and needs a constructor and a
    forward pass. The train, test and evaluation methods are implemented
    in the evaluation module with the Evaluation class.
    """

    def __init__(
        self, dim: int, num_features: int, num_classes: int, heads: int = 16
    ) -> None:
        super(Net, self).__init__(
            dim=dim, num_features=num_features, num_classes=num_classes
        )
        self.heads = heads
        self.conv3 = TransformerConv(self.num_features, self.dim,
                             dropout=0.6, heads=self.heads)

        self.fc1 = Linear(self.dim*self.heads, self.dim)
        self.fc2 = Linear(self.dim, self.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
