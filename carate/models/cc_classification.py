"""
cgc_classification model is named after the structure of the graph neural network.
The graph neural network is structured with a convolutional , graph attention,
and another convolutional layer. The cgc_classificatin model was the model
tested int the publication Introducing CARATE: Finally speaking chemistry.

:author: Julian M. Kleber
"""
from typing import Any
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
        self, dim: int, num_features: int, num_classes: int, *args, **kwargs
    ) -> None:
        super(Net, self).__init__(
            self, dim=dim, num_classes=num_classes, num_features=num_features
        )

        self.conv1 = GraphConv(self.num_features, self.dim)
        self.conv2 = GraphConv(self.dim * 16, self.dim)

        self.fc1 = Linear(self.dim, self.dim)
        self.fc2 = Linear(self.dim, self.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def __str__(self): 
        return "cc_classification-{self.dim}"