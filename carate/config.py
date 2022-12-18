"""
Module for serialization and deserialization of inputs. The aim is to 
keep web-first attitude, even though when using files locally. If there 
is text files then there is a need to convert them.

@author = Julian M. Kleber
"""
import torch

from carate.evaluation import evaluation, classification, regression
from carate.models import cgc_classification, cgc_regression
from carate.load_data import (
    DataLoader,
    StandardPytorchGeometricDataLoader,
    StandardDataLoaderTUDataset,
    StandardDataLoaderMoleculeNet,
)
from carate.utils.convert_to_json import convert_py_to_json

EVALUATION_MAP = {
    "regression": regression.RegressionEvaluation,
    "classification": classification.ClassificationEvaluation,
    "evaluation": evaluation.Evaluation,
}

MODEL_MAP = {"cgc_classification": cgc_classification, "cgc_regression": cgc_regression}

DATA_LOADER_MAP = {
    "StandardPyG": StandardPytorchGeometricDataLoader,
    "StandardTUD": StandardDataLoaderTUDataset,
    "StandardMolNet": StandardDataLoaderMoleculeNet,
}


class Config:
    """
    The Config class is an object representation of the configuration of the model. It aims to provide a middle layer between
    some user input and the run interface. It is also possible to use it via the web because of the method overload of the constructor.
    """

    def __init__(
        self,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        shrinkage: int,
        result_save_dir: str,
        Evaluation: type(evaluation.Evaluation),
        model,
        optimizer: str = None,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        data_loader: str = None,
        num_cv: int = 5,
        num_epoch: int = 150,
    ):

        # fill with maps
        self.model = model
        self.optimizer = optimizer
        self.Evaluation = Evaluation
        self.DataLoader = data_loader

        # model parameters
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_features = num_features
        self.shrinkage = shrinkage
        self.net_dimension = net_dimension
        self.learning_rate = learning_rate

        # evuluation parameters
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_cv = num_cv
        self.num_epoch = num_epoch
        self.result_save_dir = result_save_dir

    @classmethod
    def from_file(cls, file_name: str) -> None:

        json_object = convert_py_to_json(file_name)
        config_object = Config.from_json(json_object)
        return config_object

    @classmethod
    def from_json(cls, json_object: dict):
        """
        The __initialize function is a helper function that initializes the class with the parameters
        specified in the json_object. The json_object is a dictionary of key-value pairs, where each key
        is one of the following: model, optimizer, evaluation, data loader. Each value is another dictionary
        of key-value pairs specific to that particular object.

        :param self: Used to Represent the instance of the class.
        :param json_object:dict: Used to pass in the json object that is read from the config file.
        :return: A dictionary of the parameters for each class.

        :doc-author: Julian M. Kleber
        """
        return cls(
            model=MODEL_MAP[json_object["model"]],
            optimizer=json_object["optimizer"],
            Evaluation=EVALUATION_MAP[json_object["evaluation"]],
            data_loader=DATA_LOADER_MAP[json_object["data_loader"]],
            # model parameters
            dataset_name=json_object["dataset_name"],
            num_classes=int(json_object["num_classes"]),
            num_features=int(json_object["num_features"]),
            shrinkage=int(json_object["shrinkage"]),
            net_dimension=int(json_object["net_dimension"]),
            learning_rate=float(json_object["learning_rate"]),
            # evaluation parameters
            dataset_save_path=json_object["dataset_save_path"],
            test_ratio=int(json_object["test_ratio"]),
            batch_size=int(json_object["batch_size"]),
            shuffle=bool(json_object["shuffle"]),
            num_cv=int(json_object["num_cv"]),
            num_epoch=int(json_object["num_epoch"]),
            result_save_dir=json_object["result_save_dir"],
        )
