import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool, GraphConv, GATConv



import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
import sklearn.metrics as metrics
from torch_geometric.nn import global_add_pool, GraphConv, GATConv

class Net(torch.nn.Module):
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

    def train(epoch):
        model.train()

        if epoch == 51:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr']

        correct = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = F.one_hot(data.y, num_classes = 6).type(torch.FloatTensor)
            #assert len(data.y) == len(train_loader.dataset), (str(len(train_loader.dataset))+" "+str(len(data.y)))
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            loss = torch.nn.BCELoss()
            loss = loss(output_probs, data.y)
            loss.backward()
            optimizer.step()
            correct += (output == data.y).float().sum()/6
        return correct / len(train_loader.dataset)


    def test(self, loader, epoch, test=False):
        self.eval()

        correct = 0
        if test: 
        outs = []
        for data in loader:
            data.x = data.x.type(torch.FloatTensor)
            #data.y = F.one_hot(data.y, num_classes = 6).type(torch.FloatTensor)
            data = data.to(device)
            output_probs = model(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            correct += (torch.argmax(output, dim=1) == data.y).float().sum()
            #print(output, data.y)
            if test: 
            outs.append(output.cpu().detach().numpy())
        #if test: 
        #outputs =np.concatenate(outs, axis=0 ).astype(float)
        #np.savetxt("Enzymes_epoch"+str(epoch)+".csv", outputs)
        return correct / len(loader.dataset)




       