"""
File for data loading from the standard datasets implemented in the pytorch_geometric #
library
"""

from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import rdkit as rdkit

class DataLoader(): 

    def __init__(self): 
        pass

    def load(self):
        pass 


def StandardDataLoader(DataLoader): 

    #TODO integrate standard data loading 

    def __init__(self): 

    def load_standard_datset(
        path: str, dataset_name: str, test_ratio: int, batch_size: int = 64
    ):
        """
        The load_dataset function loads the SIDER dataset, splits it into a training and testing set,
        and returns the appropriate dataloaders for each. The test_ratio parameter specifies what percentage of
        the original dataset should be used as the testing set. The batch_size parameter specifies how many samples
        should be in each batch.

        :param path:str: Used to Define the path where the dataset is located.
        :param dataset_name:str: Used to Specify which dataset to load.
        :param test_ratio:int: Used to divide the dataset into a training and test set.
        :param batch_size:int: Used to set the batch size for training.
        :return: A train_loader and a test_loader.

        :doc-author: Julian M. Kleber
        """

        path = "."
        dataset = MoleculeNet(path, name="sider").shuffle()
        test_dataset = dataset[: len(dataset) // test_ratio]
        train_dataset = dataset[len(dataset) // test_ratio :]
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        return train_loader, test_loader
