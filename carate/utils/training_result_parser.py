"""
Module for parsing training results

:author: Julian M. Kleber
"""
from typing import Type, Dict, Any, List

from amarium.utils import load_json_from_file


def load_training_from_json_file(file_path: str) -> Dict[Any, Any]:
    """
    The load_training_from_json_file function loads a training result from a JSON file.


    :param file_path:str: Used to Specify the path to the file that you want to load.
    :return: A dictionary.

    :doc-author: Trelent
    """

    training_result: Dict[Any, Any] = load_json_from_file(file_path=file_path)
    return training_result


def get_loss_json(json_object: Dict[Any, List[float]]) -> List[float]:
    """
    The get_loss_json function takes in a json object and returns the loss value from that json object.

    :param json_object:dict: Used to Specify that the function takes a dictionary as an argument.
    :return: A list of dictionaries.

    :doc-author: Trelent
    """

    loss = json_object["Loss"]
    return loss


def get_accuracy(json_object: Dict[Any, List[float]]) -> List[float]:
    """
    The get_accuracy function takes in a json object and returns the accuracy of the model.
        Args:
            json_object (dict): A dictionary containing all of the information from a single run.

        Returns:
            List[float]: The accuracy for each epoch during training.

    :param json_object:dict: Used to Specify the type of the parameter.
    :return: A list of floats.

    :doc-author: Trelent
    """

    acc = json_object["Acc"]
    return acc


def get_auc(json_object: Dict[Any, List[float]]) -> List[float]:

    auc = json_object["AUC"]
    return auc
