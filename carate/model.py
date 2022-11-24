import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool, GraphConv, GATConv

import numpy as np
import torch

import sklearn.metrics as metrics


class Net(torch.nn.Module):
    """
    The Net is the core algorithm and needs a constructor and a 
    forward pass. The train, test and evaluation methods are implemented
    in the evaluation module with the Evaluation class.
    
    """
    def __init__(self, dim):
        super(Net, self).__init__()

        num_features = dataset.num_features
        self.dim = dim

        self.conv1 = GraphConv(num_features, dim)
        #self.conv2 = GraphConv(dim, dim)
        self.conv3 = GATConv(dim, dim, dropout = 0.6, heads=16)
        self.conv5 = GraphConv(dim*16, dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        #x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
