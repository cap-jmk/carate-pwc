import torch
import click

from carate.models.cgc_classification import Net
from carate.load_data import DataLoader
from carate.evaluation.evaluation import Evaluation
from carate.default_interface import DefaultObject
from carate.config import Config
from carate.optimizer import get_optimizer
from typing import Type

import logging

logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


# TODO add options for the other parameters to omit config file if wanted


class Run(DefaultObject):
    """
    Run module to parametrize different tests and benchmarks from the command line
    """

    def __init__(
        self,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        gamma: int,
        result_save_dir: str,
        model_save_freq: float,
        Evaluation: type(Evaluation),
        model_net: type(torch.nn.Module),
        optimizer: type(torch.optim),
        device: type(torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        DataLoader: type(DataLoader) = None,
        num_cv: int = 5,
        num_epoch=150,
    ):
        # model parameters
        self.dataset_name = dataset_name
        self.device = device
        self.num_classes = num_classes
        self.num_features = num_features
        self.gamma = gamma
        self.Evaluation = Evaluation
        self.model_net = model_net
        self.net_dimension = net_dimension
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.model_save_freq = model_save_freq
        # evaulation parameters
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_cv = num_cv
        self.num_epoch = num_epoch
        self.result_save_dir = result_save_dir

        self.DataLoader = DataLoader

    @classmethod
    def from_file(cls, config_filepath: str) -> None:

        config = Config.from_file(file_name=config_filepath)
        run_object = Run.__init_config(config)
        return run_object

    @classmethod
    def from_json(cls, json_object: dict) -> None:

        config = Config.from_json(json_object=json_object)
        run_object = Run.__init_config(config)
        return run_object

    def run(
        self,
        device: type(torch.optim) = None,
        dataset_name: str = None,
        test_ratio: int = None,
        dataset_save_path: str = None,
        model_net: type(torch.nn.Module) = None,
        optimizer: type(torch.optim) = None,
        DataLoader: type(DataLoader) = None,
        num_cv: int = None,
        num_epoch: int = None,
        num_classes: int = None,
        num_features: int = None,
        batch_size: int = None,
        shuffle: int = None,
        gamma: int = None,
        result_save_dir: str = None,
        Evaluation: type(Evaluation) = None,
        model_save_freq: type(int) = None,
    ) -> None:

        (
            device,
            dataset_name,
            test_ratio,
            dataset_save_path,
            model_net,
            optimizer,
            DataLoader,
            num_cv,
            num_epoch,
            num_classes,
            num_features,
            batch_size,
            shuffle,
            gamma,
            result_save_dir,
            Evaluation,
            model_save_freq,
        ) = self._get_defaults(locals())

        self.Evaluation = Evaluation(
            dataset_name=dataset_name,
            dataset_save_path=dataset_save_path,
            test_ratio=test_ratio,
            model_net=model_net,
            optimizer=optimizer,
            DataLoader=DataLoader,
            device=device,
            gamma=gamma,
            result_save_dir=result_save_dir,
            model_save_freq=model_save_freq,
        )

        self.Evaluation.cv(
            dataset_name=dataset_name,
            dataset_save_path=dataset_save_path,
            test_ratio=test_ratio,
            num_cv=num_cv,
            num_epoch=num_epoch,
            num_classes=num_classes,
            DataLoader=DataLoader,
            shuffle=shuffle,
            batch_size=batch_size,
            model_net=model_net,
            optimizer=optimizer,
            device=device,
            gamma=gamma,
            result_save_dir=result_save_dir,
            model_save_freq=model_save_freq,
        )

    @classmethod
    def __init_config(cls, config: type(Config)) -> None:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_net = config.model.Net(
            dim=int(config.net_dimension),
            num_classes=int(config.num_classes),
            num_features=int(config.num_features),
        ).to(device)
        optimizer = get_optimizer(
            optimizer_str=config.optimizer,
            model_net=model_net,
            learning_rate=config.learning_rate,
        )
        return cls(
            dataset_name=config.dataset_name,
            device=device,
            num_classes=config.num_classes,
            num_features=config.num_features,
            gamma=config.gamma,
            Evaluation=config.Evaluation,
            model_net=model_net,
            optimizer=optimizer,
            net_dimension=config.net_dimension,
            learning_rate=config.learning_rate,
            # evaulation parameters
            dataset_save_path=config.dataset_save_path,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_cv=config.num_cv,
            num_epoch=config.num_epoch,
            result_save_dir=config.result_save_dir,
            DataLoader=config.DataLoader(
                dataset_save_path=config.dataset_save_path,
                dataset_name=config.dataset_name,
                test_ratio=config.test_ratio,
                batch_size=config.batch_size,
            ),
            model_save_freq=int(config.model_save_freq),
        )
