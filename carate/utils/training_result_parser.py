"""
Module for parsing training results 

:author: Julian M. Kleber
"""
from carate.utils import load_json_file


def load_training_from_json_file(file_path: str) -> dict:

    training_result = load_json_file(file_path=file_path)
    return training_result


def get_loss_json(json_object: dict) -> list:

    loss = json_object["Loss"]
    return loss


def get_accuracy(json_object: dict) -> list:

    acc = json_object["Acc"]
    return acc


def get_auc(json_object: dict) -> list:

    auc = json_object["AUC"]
    return auc
