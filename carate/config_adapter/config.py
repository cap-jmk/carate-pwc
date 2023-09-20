"""
Module for serialization and deserialization of inputs. The aim is to
keep web-first attitude, even though when using files locally. If there
is text files then there is a need to convert them.

@author = Julian M. Kleber
"""
import torch
from typing import Type, Optional, Dict, TypeVar, Any, Generic

from amarium.utils import convert_str_to_bool

from carate.evaluation import base, classification, regression
from carate.models import (
    cgc_classification,
    cgc_regression,
    g_classification,
    g_regression,
    cc_classification,
    cc_regression,
    gcc_classification,
    gcc_regression,
    gatv2_classification,
    gatv2_regression,
    graph_transformer_classification,
    graph_transformer_regression,
)

from carate.loader.load_data import (
    DatasetObject,
    StandardPytorchGeometricDataset,
    StandardDatasetTUDataset,
    StandardDatasetMoleculeNet,
)
from carate.utils.convert_to_json import convert_py_to_json
from carate.logging.metrics_logger import MetricsLogger

EvaluationMap: Dict[str, base.Evaluation]
EVALUATION_MAP = {
    "regression": regression.RegressionEvaluation,
    "classification": classification.ClassificationEvaluation,
    "evaluation": base.Evaluation,
}

ModelMap: Dict[str, Any]
MODEL_MAP = {
    "cgc_classification": cgc_classification,
    "cgc_regression": cgc_regression,
    "g_classification": g_classification,
    "g_regression": g_regression,
    "cc_classification": cc_classification,
    "cc_regression": cc_regression,
    "gatv2_classification": gatv2_classification,
    "gatv2_regression": gatv2_regression,
    "graph_transformer_classification": graph_transformer_classification,
    "graph_transformer_regression": graph_transformer_regression,
}

DATA_SET_MAP: Dict[
    str,
    Type[StandardDatasetMoleculeNet]
    | Type[StandardPytorchGeometricDataset]
    | Type[StandardPytorchGeometricDataset],
]
DATA_SET_MAP = {
    "StandardPyG": StandardPytorchGeometricDataset,
    "StandardTUD": StandardDatasetTUDataset,
    "StandardMolNet": StandardDatasetMoleculeNet,
}


class Config:
    """
    The Config class is an object representation of the configuration of the model. It aims to provide a middle layer between
    some user input and the run interface. It is also possible to use it via the web because of the method overload of the constructor.

    :author: Julian M. Kleber
    """

    def __init__(
        self,
        file_path:str,
        dataset_name: str,
        num_features: int,
        num_classes: int,
        result_save_dir: str,
        model_save_freq: int,
        Evaluation: base.Evaluation,
        data_set: DatasetObject,
        model: Any,
        logger: Any, 
        optimizer: str,
        device: str = "auto",
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        num_cv: int = 5,
        num_epoch: int = 150,
        override: bool = True,
        resume: bool = False,
        normalize: bool = False,
        num_heads: int = 3,
        dropout_gat: float = 0.6,
        dropout_forward: float = 0.5,
        custom_size: Optional[int] = None,
    ):

        #save path
        self.file_path = file_path
        # modelling

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.Evaluation = Evaluation
        self.data_set = data_set
        self.normalize = normalize

        # model parameters

        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_features = num_features
        self.net_dimension = net_dimension
        self.num_heads = num_heads
        self.dropout_gat = dropout_gat
        self.dropout_forward = dropout_forward

        # evaluation parameters

        self.result_save_dir = result_save_dir
        self.model_save_freq = model_save_freq
        self.override = override

        # training
        self.resume = resume
        self.learning_rate = learning_rate
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.custom_size = custom_size
        self.num_cv = num_cv
        self.num_epoch = num_epoch

        # data
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.shuffle = shuffle
        self.logger = logger


class ConfigInitializer:
    @classmethod
    def from_file(cls, file_name: str) -> Config:
        """
        The from_file function takes a file name as an argument and returns a Config object.
        The function reads the file, converts it to JSON, then uses the from_json method to create
        the Config object.

        :param cls: Used to create a new instance of the class.
        :param file_name:str: Used to specify the name of the file to be used.
        :return: A config object.

        :doc-author: Julian M. Kleber
        """

        json_object = convert_py_to_json(file_name)
        config_object = ConfigInitializer.from_json(file_name=file_name, json_object = json_object)
        return config_object

    @classmethod
    def from_json(cls, file_name:str, json_object: Dict[Any, Any]) -> Config:
        """
        The from_json function is a class method that takes in a json object and returns an instance of the Config class.
        The function is used to load the configuration from a file, which can be done by calling:
            config = Config.from_json(json_object)

        :param cls: Used to Create an instance of the class that is calling this method.
        :param json_object:dict: Used to Pass in the json object that is read from the file.
        :return: A class object.

        :doc-author: Julian M. Kleber
        """

        if json_object["device"] == "cpu":
            device = torch.device("cpu")
        elif json_object["device"] == "cuda":
            device = torch.device("cuda")
        elif json_object["device"] == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "resume" in list(json_object.keys()):
            resume = json_object["resume"]
        else:
            resume = False

        if "normalize" in json_object.keys():
            normalize = json_object["normalize"]
        else:
            normalize = False

        if "custom_size" in json_object.keys():
            custom_size = json_object["custom_size"]
        else:
            custom_size = None

        if "num_heads" in json_object.keys():
            num_heads = json_object["num_heads"]
        else:
            num_heads = 3

        if "dropout_forward" in json_object.keys():
            dropout_forward = json_object["dropout_forward"]
        else:
            dropout_forward = 3

        if "dropout_gat" in json_object.keys():
            dropout_gat = json_object["dropout_gat"]
        else:
            dropout_gat = 0.6
        
        if "log_save_dir" not in json_object.keys(): 
            log_save_dir = json_object["result_save_dir"]
        else: 
            log_save_dir = json_object["log_save_dir"]
        
        metrics_logger = MetricsLogger(json_object["result_save_dir"])
        metrics_logger.logger.info("Initializing configuration for the config file "+ file_name)
        metrics_logger.logger.info("The configuration is: " + str(json_object))
        data_set = DATA_SET_MAP[json_object["data_set"]](
            dataset_save_path=json_object["dataset_save_path"],
            dataset_name=json_object["dataset_name"],
            test_ratio=json_object["test_ratio"],
            batch_size=json_object["batch_size"],
            shuffle=json_object["shuffle"],
        )



        evaluation = EVALUATION_MAP[json_object["evaluation"]](
            dataset_name=json_object["dataset_name"],
            dataset_save_path=json_object["dataset_save_path"],
            test_ratio=json_object["test_ratio"],
            model_net=json_object["model"],
            optimizer=json_object["optimizer"],
            data_set=data_set,
            result_save_dir=json_object["result_save_dir"],
            model_save_freq=json_object["model_save_freq"],
            device=device,
            resume=resume,
            logger = metrics_logger
        )
        json_object["override"] = convert_str_to_bool(json_object["override"])

        return Config(
            file_path = file_name,
            model=MODEL_MAP[json_object["model"]],
            optimizer=json_object["optimizer"],
            device=device,
            Evaluation=evaluation,
            data_set=data_set,
            # model parameters
            dataset_name=str(json_object["dataset_name"]),
            num_classes=int(json_object["num_classes"]),
            num_features=int(json_object["num_features"]),
            net_dimension=int(json_object["net_dimension"]),
            learning_rate=float(json_object["learning_rate"]),
            # evaluation parameters
            dataset_save_path=str(json_object["dataset_save_path"]),
            test_ratio=int(json_object["test_ratio"]),
            batch_size=int(json_object["batch_size"]),
            shuffle=bool(json_object["shuffle"]),
            num_cv=int(json_object["num_cv"]),
            num_epoch=int(json_object["num_epoch"]),
            result_save_dir=str(json_object["result_save_dir"]),
            model_save_freq=int(json_object["model_save_freq"]),
            override=json_object["override"],
            resume=resume,
            normalize=normalize,
            num_heads=num_heads,
            dropout_forward=dropout_forward,
            dropout_gat=dropout_gat,
            custom_size=custom_size,
            logger = metrics_logger
        )
