"""
File for data loading from the standard datasets implemented in the pytorch_geometric #
library. The DataSet loader is implemented as a base clase and other subclasses include loaders for standardized benchmarks
as well as custom datasets. 
"""

from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import rdkit as rdkit

class DataLoader(): 

    def __init__(self): 
        pass

    def load(self):
        pass 


class StandardDataLoader(DataLoader): 
    

    

    def __init__(self, path:str, data_set_name:str, test_ratio:int, batch_size:int):
        """
        The __init__ function is called the constructor and is automatically called when you create a new instance of this class.
        The __init__ function allows us to set attributes that are specific to each object created from the class.
        In our case, we want each data_set object to have a path, data_set_name, test_ratio and batch size attribute.
        
        :param self: Used to Reference the object to which the function is applied.
        :param path:str: Used to Specify the path to the dataset.
        :param data_set_name:str: Used to Store the name of the data set.
        :param test_ratio:int: Used to Split the data set into a training and testing set.
        :param batch_size:int: Used to Set the batch size.
        :return: The object of the class.
        
        :doc-author: Julian M. Kleber
        """
        
        
        self.path = path 
        self.data_set_name = data_set_name
        self.test_ratio = test_ratio
        self.batch_size = batch_size


    def load_standard_datset(
        path: str, dataset_name: str, test_ratio: int, batch_size: int = 64, shuffle:bool = True
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
        #TODO implement shuffle on request 

        path = "."
        if shuffle:
            dataset = MoleculeNet(path, name=dataset_name).shuffle()
        else: 
            dataset = MoleculeNet(path, name=dataset_name).shuffle()
        test_dataset = dataset[: len(dataset) // test_ratio]
        train_dataset = dataset[len(dataset) // test_ratio :]
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        return train_loader, test_loader
