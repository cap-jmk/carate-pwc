"""
Evaulation object for classification
"""
from typing import Type

import torch
from carate.evaluation.base import Evaluation
from carate.load_data import DatasetObject


class ClassificationEvaluation(Evaluation):
    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        result_save_dir: str,
        model_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_set: DatasetObject,
        model_save_freq: int,
        test_ratio: int,
        num_epoch: int = 150,
        num_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        batch_size: int = 64,
        shuffle: bool = True,
        override: bool = True,
    ):
        """

        :param self: Used to Refer to the object instance itself, and is used to access variables that belongs to the class.
        :param model: Used to Specify the model that will be trained.
        :param optimizer: Used to Define the optimizer that will be used to train the model.
        :param data_set:Type[DatasetObject]: Used to Specify the type of data loader that is used. Is implemented according to
                                             the interface given in load_data.py by the class DatasetObject.load_data().

        :param epoch:int=150: Used to Set the number of epochs to train for.
        :param num_cv:int=5: Used to Specify the number of cross validations that will be used in the training process.
        :param num_classes:int=2: Used to Define the number of classes in the dataset.
        :param out_dir:str="out": Used to Specify the directory where the output of your training will be stored.
        :return: The following:.

        :doc-author: Julian M. Kleber
        """
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.num_epoch = num_epoch
        self.model_net = model_net
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.num_cv = num_cv
        self.out_dir = out_dir
        self.data_set = data_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_save_dir = result_save_dir
        self.model_save_freq = model_save_freq
        self.override = override

    def __repr__(self) -> str:
        return "Classification Evaluation Object"
