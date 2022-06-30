import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, num_node_features: int, num_nodes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_nodes)
        self.conv2 = GCNConv(num_nodes, num_nodes)
        self.fc = torch.nn.Linear(num_nodes*num_nodes, num_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.flatten(x)
        return self.fc(x)

class Data:
    feature: list
    target: list

class DatasetIterator:
    def __init__(self, features, targets, batch_size=64):
        self.__batch_size = batch_size
        self.__features = features
        self.__targets = targets
        self.__i = 0
        self.__numbers = 0
        self.__random_index = list(range(len(self.__features)))
        random.shuffle(self.__random_index)

    @property
    def num_batch(self):
        return len(self.__features) // self.__batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.__i == self.num_batch:
            idx_list = self.__random_index[self.__i * self.__batch_size:]
            self.__i += 1
            return Data(
                feature=self.__features[idx_list],
                target=self.__targets[idx_list],
            )
        if self.__i == (self.num_batch + 1):
            raise StopIteration()
        idx_list = self.__random_index[self.__i * self.__batch_size:(self.__i + 1) * self.__batch_size]
        self.__i += 1
        return Data(
                feature=self.__features[idx_list],
                target=self.__targets[idx_list],
            )
    

class Dataset:
    def __init__(self, batch_size=64):
        self.__batch_size = batch_size
        self.__features = []
        self.__targets = []

    def add_feature(self, feature):
        self.__features.append(feature)

    def add_target(self, target):
        self.__targets.append(target)

    @property
    def num_batch(self):
        return len(self.__features) // self.__batch_size
        
    def __iter__(self):
        return DatasetIterator(np.array(self.__features), np.array(self.__targets.copy()), self.__batch_size)
