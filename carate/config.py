"""
Module for serialization and deserialization of inputs. The aim is to 
keep web-first attitude, even though when using files locally. If there 
is text files then there is a need to convert them.
#TODO provide a utility for it 

@author = Julian M. Kleber
"""
from carate.evaluation import evaluation, classification, regression
from carate.models import cgc_classification, cgc_regression
from carate.load_data import DataLoader, StandardPytorchGeometricDataLoader, StandardDataLoaderTUDataset, StandardDataLoaderMoleculeNet


EVALUATION_MAP = {
    "regression" : regression.RegressionEvaluation,
    "classification": classification.ClassificationEvaluation,
    "evaluation": evaluation.Evaluation
}

MODEL_MAP = {
    "cgc_classification" : cgc_classification, 
    "cgc_regression" : cgc_regression
}

DATA_LOADER_MAP = {
    "StandardPyG" : StandardPytorchGeometricDataLoader,
    "StandardTUD" : StandardDataLoaderTUDataset,
    "StandardMolNet": StandardDataLoaderMoleculeNet
}

OPTIMIZER_MAP = {
    "adams" : torch.optim.Adam(
                self.model_net.parameters(), lr=learning_rate
            )
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
        evaluation: str,
        model: str,
        optimizer: str= None,
        net_dimension: int = 364,
        learning_rate: float = 0.0005,
        dataset_save_path: str = ".",
        test_ratio: int = 20,
        batch_size: int = 64,
        shuffle: bool = True,
        data_loader: str = None,
        n_cv: int = 5,
        num_epoch:int =150,
    ):

        # fill with maps 
        self.model = MODEL_MAP[model]
        self.optimizer = OPTIMIZER_MAP[optimizer]
        self.Evaluation = EVALUATION_MAP[evaluation]
        self.DataLoader = DATA_LOADER_MAP[data_loader]
        # model parameters
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_features = num_features
        self.shrinkage = shrinkage
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
    

    @classmethod()
    def __init__(self, json_object:dict=None)->None: 

        self.__initialize(json_object)
    
    @classmethod()
    def __init_(self, file_name:str)->None:
        json_object = deserialization(file_name)
        self.__initialize(json_object)
    
    def __initialize(json_object:dict): 
        