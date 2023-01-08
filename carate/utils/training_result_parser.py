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

    training_result = load_json_from_file(file_path=file_path)
    training_result: Dict[Any, Any]
    return training_result


def get_loss_json(json_object: dict) -> List[Any]:

    loss = json_object["Loss"]
    return loss


def get_accuracy(json_object: dict) -> list:

    acc = json_object["Acc"]
    return acc


def get_auc(json_object: dict) -> list:

    auc = json_object["AUC"]
    return auc
