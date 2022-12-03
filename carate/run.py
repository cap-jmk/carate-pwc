import torch
import click

from carate.models.cgc import Net
from carate.load_data import DataLoader, StandardDataLoader
from carate.evaluation import Evaluation
from carate.default_interface import DefaultObject

from typing import Type

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class Run(DefaultObject):
    """
    Run module to parametrize different tests and benchmarks from the command line
    """

    def __init__(
        self,
        data_set_name: str,
        num_features: int,
        num_classes: int,
        model: type(torch.nn.Module),
        device: type(torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        optimizer: type(torch.optim) = None,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        data_set_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        DataLoader: type(DataLoader) = None,
        n_cv: int = 5,
        num_epoch=150,
    ):
        # model parameters
        self.device = device
        self.num_classes = num_classes
        self.num_features = num_features
        self.model = model(
            dim=net_dimension, num_classes=num_classes, num_features=num_features
        ).to(device)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.net_dimension = net_dimension
        self.learning_rate = learning_rate

        # evaulation parameters
        self.data_set_name = data_set_name
        self.data_set_save_path = data_set_save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.n_cv = n_cv
        self.num_epoch = num_epoch
        if DataLoader is None:
            self.DataLoader = StandardDataLoader(
                path=self.data_set_save_path,
                data_set_name=self.data_set_name,
                test_ratio=self.test_ratio,
                batch_size=self.batch_size,
            )

    def run(
        self,
        model: type(torch.nn.Module) = None,
        optimizer: type(torch.optim) = None,
        DataLoader: type(DataLoader) = None,
        n_cv: int = None,
        num_epoch: int = None,
        num_classes: int = None,
        num_features: int = None,
    ):

        (
            model,
            optimizer,
            DataLoader,
            n_cv,
            num_epoch,
            num_classes,
            num_features,
        ) = self._get_defaults(locals()) # TODO Error got multiple values for n_cv it may be that the arguments are recieved in the wrong order or something
        self.Evaluation = Evaluation(
            model=model, optimizer=optimizer, DataLoader=DataLoader
        )
        self.Evaluation.cv(
            n_cv=n_cv,
            num_epoch=num_epoch,
            num_classes=num_classes,
            num_features=num_features,
            DataLoader=DataLoader,
        )


if __name__ == "__main__":
    pass
