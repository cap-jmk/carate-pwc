from typing import Type, Dict, Any, TypeVar, Generic
import torch


from carate.models.base_model import Model
from carate.models.cgc_classification import Net
from carate.load_data import DatasetObject
from carate.evaluation.base import Evaluation
from carate.default_interface import DefaultObject
from carate.config import ConfigInitializer, Config
from carate.optimizer import get_optimizer
from typing import Type, Optional

import logging

logging.basicConfig(
    filename="train.log",
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
        dataset_name: str,
        num_features: int,
        num_classes: int,
        result_save_dir: str,
        model_save_freq: int,
        data_set: DatasetObject,
        Evaluation: Evaluation,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        override: bool,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        num_cv: int = 5,
        num_epoch: int = 150,
    ) -> None:
        """
        Constructor
        """
        # model parameters
        self.dataset_name = dataset_name
        self.device = device
        self.num_classes = num_classes
        self.num_features = num_features
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

        self.data_set = data_set
        self.override = override

    def run(self) -> None:

        self.Evaluation.cv(
            dataset_name=self.dataset_name,
            dataset_save_path=self.dataset_save_path,
            test_ratio=self.test_ratio,
            num_cv=self.num_cv,
            num_epoch=self.num_epoch,
            num_classes=self.num_classes,
            data_set=self.data_set,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            model_net=self.model_net,
            optimizer=self.optimizer,
            device=self.device,
            result_save_dir=self.result_save_dir,
            model_save_freq=int(self.model_save_freq),
            override=self.override,
        )


class RunInitializer:
    @classmethod
    def from_file(cls, config_filepath: str) -> Run:

        config = ConfigInitializer.from_file(file_name=config_filepath)
        run_object = RunInitializer.__init_config(config)
        return run_object

    @classmethod
    def from_json(cls, json_object: Dict[Any, Any]) -> Run:

        config = ConfigInitializer.from_json(json_object=json_object)
        run_object = RunInitializer.__init_config(config)
        return run_object

    @classmethod
    def __init_config(cls, config: Config) -> Run:
        """
        The __init_config function initializes the configuration of the model.


        :param self: Used to Represent the instance of the class.
        :param config:type(Config): Used to Pass in the config class.
        :return: None.

        :doc-author: Julian M. Kleber
        """
       
        model_net = config.model.Net(
            dim=int(config.net_dimension),
            num_classes=int(config.num_classes),
            num_features=int(config.num_features),
        ).to(config.device)
        optimizer = get_optimizer(
            optimizer_str=config.optimizer,
            model_net=model_net,
            learning_rate=config.learning_rate,
        )
        return Run(
            dataset_name=config.dataset_name,
            device=config.device,
            num_classes=config.num_classes,
            num_features=config.num_features,
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
            data_set=config.data_set,
            model_save_freq=int(config.model_save_freq),
            override=config.override,
        )
