import torch
from models.cgc import Net
from cate.load_data import StandardDataLoader
import click


from typing import Type


class Run:
    """
    Run module to parametrize different tests and benchmarks from the command line
    """

    def __init__(
        self,
        data_set_name: str,
        model: Type(torch.nn.Module),
        device: Type(torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),

        optimizer: Type(torch.optim) = None,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        data_set_save_path: str = ".",
        
        test_ratio: int = 20,
        batch_size: int = 64,
    ):
        # model parameters
        self.device = device
        self.model = model(dim=net_dimension).to(device)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.net_dimension = 364
        self.learning_rate = 0.0005

        # evaulation parameters
        self.data_set_name = data_set_name
        self.data_set_save_path = data_set_save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        

    def run(self):
        pass
        


    def load_data(self):
        """
        The load_data function loads the data set, and returns a train loader and test loader.
        The train_loader is used to load training data in batches for model training. The test_loader is 
        used to load testing data in batches for model evaluation.
        
        :param self: Used to Access variables that belongs to the class.
        :return: A train_loader and a test_loader.
        
        :doc-author: Trelent
        """
        self.StandardDataLoader(path=self.data_set_save_path,
            dataset_name=self.data_set_name,
            test_ratio=self.test_ratio,
            batch_size=self.batch_size)
        self.train_loader, self.test_loader = load_standard_datset(
            path=self.data_set_save_path,
            dataset_name=self.data_set_name,
            test_ratio=self.test_ratio,
            batch_size=self.batch_size,
        )
        return self.train_loader, self.test_loader

if __name__ == "__main__":
    run = Run()
    print(run.model.dim)
    run.net_dimension = 50
    print(run.model.dim)
    run.__init__(net_dimension=50)
    print(run.model.dim)
