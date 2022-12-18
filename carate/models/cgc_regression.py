"""
cgc_regression model is named after the structure of the graph neural network. 
The graph neural network is structured with a convolutional , graph attention, 
and another convolutional layer. The cgc_classificatin model was the model tested int the publication 
Introducing CARATE: Finally speaking chemistry.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool, GraphConv, GATConv

import numpy as np
import sklearn.metrics as metrics

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class Net(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int, num_features: int) -> None:
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.dim = dim
        self.conv1 = GraphConv(self.num_features, self.dim)
        self.conv3 = GATConv(self.dim, self.dim, dropout=0.6)
        self.conv5 = GraphConv(self.dim, self.dim)

        self.fc1 = Linear(self.dim, self.dim)
        self.fc2 = Linear(self.dim, self.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
