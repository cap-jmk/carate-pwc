import os 

from typing import Type, Dict, Any, TypeVar, Generic
import torch

from amarium.utils import make_full_filename, read_file, write_file

from carate.models.base_model import Model
from carate.models.cgc_classification import Net
from carate.loader.load_data import DatasetObject
from carate.evaluation.base import Evaluation
from carate.default_interface import DefaultObject
from carate.config_adapter.config import ConfigInitializer, Config
from carate.optimizer.optimizer import get_optimizer
from typing import Type, Optional

import logging

logging.basicConfig(
    filename="carate.log",
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
        resume: bool,
        normalize: bool,
        data_set: DatasetObject,
        Evaluation: Evaluation,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        override: bool,
        logger: Any, 
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        num_cv: int = 5,
        num_epoch: int = 150,
        num_heads: int = 3,
        dropout_forward: float = 0.6,
        dropout_gat: float = 0.5,
        custom_size: Optional[int] = None,
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
        self.optimizer = optimizer
        self.num_heads = num_heads
        self.dropout_gat = dropout_gat
        self.dropout_forward = dropout_forward
        self.dropout_gat = dropout_gat

        # evaulation / training parameters
        self.model_save_freq = model_save_freq
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.num_cv = num_cv
        self.num_epoch = num_epoch

        # data set
        self.data_set = data_set
        self.override = override
        self.resume = resume
        self.normalize = normalize
        self.custom_size = custom_size
        self.result_save_dir = result_save_dir
        self.dataset_name = dataset_name

        # Results
        self.dataset_save_path = dataset_save_path
        self.logger = logger

    def run(self) -> None:
        """
        Function to run training a model. Here only the CV is considered 
        #TODO Make it more flexibile by passing the function as a parameter
        """
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
            resume=self.resume,
            custom_size=self.custom_size,
            logger = self.logger
        )
        
        self.logger.close_logger() #close the current logging file after a run




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

        #copy config to result dir


        # Model initalization
        model_net = config.model.Net(
            dim=int(config.net_dimension),
            num_classes=int(config.num_classes),
            num_features=int(config.num_features),
            num_heads=int(config.num_heads),
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
            resume=config.resume,
            normalize=config.normalize,
            custom_size=config.custom_size,
            logger = config.logger
        )

