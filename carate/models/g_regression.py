import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool, GraphConv, GATConv

import numpy as np
import sklearn.metrics as metrics

import logging


from carate.models.base_model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="carate.log",
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
        self,
        dim: int,
        num_features: int,
        num_classes: int,
        num_heads: int = 16,
        dropout_gat: float = 0.6,
        dropout_forward: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super(Net, self).__init__(
            dim=dim, num_classes=num_classes, num_features=num_features
        )

        self.num_heads = num_heads
        self.dropout_gat = dropout_gat
        self.dropout_forward = dropout_forward

        self.dropout_forward = dropout_forward

        self.conv1 = GATConv(
            self.num_features, self.dim, dropout=self.dropout_gat, heads=self.num_heads
        )

        self.fc1 = Linear(self.dim * self.num_heads, self.dim)
        self.fc2 = Linear(self.dim, self.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout_forward, training=self.training)
        x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def __str__(self): 
        return "g_regression"