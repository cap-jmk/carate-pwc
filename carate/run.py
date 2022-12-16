import torch
import click

from carate.models.cgc_classification import Net
from carate.load_data import DataLoader, StandardDataLoaderMoleculeNet
from carate.evaluation.evaluation import Evaluation
from carate.default_interface import DefaultObject
from carate.config import Config
from typing import Type

import logging

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)



#TODO add options for the other parameters to omit config file if wanted


class Run(DefaultObject):
    """
    Run module to parametrize different tests and benchmarks from the command line
    """

    def __init__(
        self,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        shrinkage: int,
        result_save_dir: str,
        Evaluation: type(Evaluation),
        model: type(torch.nn.Module),
        device: type(torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        optimizer: type(torch.optim) = None,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        DataLoader: type(DataLoader) = None,
        n_cv: int = 5,
        num_epoch=150,
    ):
        # model parameters
        self.dataset_name = dataset_name
        self.device = device
        self.num_classes = num_classes
        self.num_features = num_features
        self.shrinkage = shrinkage
        self.Evaluation = Evaluation
        self.model_net = model.Net(
            dim=net_dimension, num_classes=num_classes, num_features=num_features
        ).to(device)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model_net.parameters(), lr=learning_rate
            )
        self.net_dimension = net_dimension
        self.learning_rate = learning_rate

        # evaulation parameters
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_cv = n_cv
        self.num_epoch = num_epoch
        self.result_save_dir = result_save_dir

        self.DataLoader = DataLoader(
            dataset_save_path=self.dataset_save_path,
            dataset_name=self.dataset_name,
            test_ratio=self.test_ratio,
            batch_size=self.batch_size,
        )

    @classmethod
    def from_file(cls, config_filepath:str)->None:

        self.config = Config.from_file(file_name=config_filepath)
        run_object = Run.from_json(config)
        return run_object
        
    @classmethod
    def from_json(cls, json_object:dict)->None:

        self.config = Config(json_object=json_object)
        run_object = Run.__init_config(config)
        return run_object


    def run(
        self,
        device: type(torch.optim),
        dataset_name: str = None,
        test_ratio: int = None,
        dataset_save_path: str = None,
        model_net: type(torch.nn.Module) = None,
        optimizer: type(torch.optim) = None,
        DataLoader: type(DataLoader) = None,
        n_cv: int = None,
        num_epoch: int = None,
        num_classes: int = None,
        num_features: int = None,
        batch_size: int = None,
        shuffle: int = None,
        shrinkage: int = None,
        result_save_dir: str = None,
        Evaluation: type(Evaluation) = None,
    )->None:

        (
            device,
            dataset_name,
            test_ratio,
            dataset_save_path,
            model_net,
            optimizer,
            DataLoader,
            n_cv,
            num_epoch,
            num_classes,
            num_features,
            batch_size,
            shuffle,
            shrinkage,
            result_save_dir,
            Evaluation,
        ) = self._get_defaults(locals())

        self.Evaluation = Evaluation(
            dataset_name=dataset_name,
            dataset_save_path=dataset_save_path,
            test_ratio=test_ratio,
            model_net=model_net,
            optimizer=optimizer,
            DataLoader=DataLoader,
            device=device,
            shrinkage=shrinkage,
            result_save_dir=result_save_dir,
        )

        self.Evaluation.cv(
            dataset_name=dataset_name,
            dataset_save_path=dataset_save_path,
            test_ratio=test_ratio,
            n_cv=n_cv,
            num_epoch=num_epoch,
            num_classes=num_classes,
            DataLoader=DataLoader,
            shuffle=shuffle,
            batch_size=batch_size,
            model_net=model_net,
            optimizer=optimizer,
            device=device,
            shrinkage=shrinkage,
            result_save_dir=result_save_dir,
        )

    def __init_config(self, config:type(Config))->None: 
        """
        The __init_config function initializes the configuration of the model.
        
        Parameters: 
            config (type(Config)): The configuration object that contains all parameters for training and evaluation.
        
            Returns: None
        
        :param self: Used to Represent the instance of the class.
        :param config:type(Config): Used to Pass in the config class.
        :return: None.
        
        :doc-author: Julian M. Kleber
        """
        if config.optimizer is None:
            optimizer = torch.optim.Adam(
                self.model_net.parameters(), lr=learning_rate
            )
        return cls(
        dataset_name = config.dataset_name,
        device = config.device,
        num_classes = config.num_classes,
        num_features = config.num_features,
        shrinkage = config.shrinkage,
        Evaluation = config.Evaluation,
        model_net = config.model.Net(
            dim=net_dimension, num_classes=num_classes, num_features=num_features
        ).to(device),
        
        optimizer = optimizer,
        net_dimension = config.net_dimension,
        learning_rate = config.learning_rate,

        # evaulation parameters
        dataset_save_path = config.dataset_save_path,
        test_ratio = config.test_ratio,
        batch_size = config.batch_size,
        shuffle = config.shuffle,
        n_cv = config.n_cv,
        num_epoch = config.num_epoch,
        result_save_dir = config.result_save_dir,

        DataLoader = config.DataLoader(
            dataset_save_path=self.dataset_save_path,
            dataset_name=self.dataset_name,
            test_ratio=self.test_ratio,
            batch_size=self.batch_size,
        )
        )

