"""
Module to handle optimizer initalization


@author Julian M. Kleber
"""
import torch
from typing import Type

from carate.models.base_model import Model


def get_optimizer(
    optimizer_str: str, model_net: Model, learning_rate: float
) -> torch.optim.Optimizer:
    """
    The get_optimizer function takes in a string and returns the corresponding optimizer.
        Args:
            optimizer_str (str): The name of the desired optimizer.

        Returns:
            Type[torch.optim.Optimizer]: The corresponding torch optimization function for the given string.

    :param optimizer_str:str: Used to Specify the type of optimizer we want to use.
    :param model: Used to Get the parameters of the model.
    :return: The optimizer of the network.

    :doc-author: Trelent
    """

    if optimizer_str == "adams":
        return torch.optim.Adam(model_net.parameters(), lr=learning_rate)
