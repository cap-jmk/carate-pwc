"""
Utility file for model checkpoints file operations

:author: Julian M. Kleber
"""
import os

import torch

from amarium.utils import (
    prepare_file_name_saving,
    load_json_from_file,
    make_full_filename,
    save_json_to_file,
    check_make_dir,
)

from typing import Tuple, Type, Dict, Any

from carate.models.base_model import Model


def load_model(
    model_path: str, model_net: Type[torch.nn.Module]
) -> Type[torch.nn.Module]:
    """
    The load_model function takes in a model_path, model_params_path and the type of network to be loaded.
    It then loads the parameters from the params file into a dictionary and uses that to create an instance of
    the specified network. It then loads in the state dict from PATH and sets it as eval mode.

    :param model_path:str: Used to specify the path to the model file.
    :param model_params_path:str: Used to load the model parameters from a file.
    :param model_net:Type[torch.nn.Module]: Used to specify the type of model that is being loaded.
    :return: A model that is loaded with the parameters in the path.

    :doc-author: Julian M. Kleber
    """

    model = model_net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_model_training_checkpoint(
    checkpoint_path: str,
    model_net: Type[torch.nn.Module],
    optimizer: Type[torch.optim.Optimizer],
) -> Tuple[Model, torch.optim.Optimizer]:

    # For any bug fixing please consult the PyTorch documentation:  https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    model = model_net
    optimizer = optimizer
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    model.train()
    return model, optimizer


def save_model_training_checkpoint(
    result_save_dir: str,
    dataset_name: str,
    num_cv: int,
    num_epoch: int,
    model_net: Type[torch.nn.Module],
    optimizer: Type[torch.optim.Optimizer],
    loss: float,
    override: bool,
) -> None:

    # For any bug fixing please refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    prefix = result_save_dir + "/checkpoints/CV_" + str(num_cv)
    check_make_dir(prefix)

    if override is True:
        save_path = prepare_file_name_saving(
            prefix=prefix,
            file_name=dataset_name,
            suffix=".tar",
        )
    if override is False:
        save_path = prepare_file_name_saving(
            prefix=prefix,
            file_name=dataset_name + "_Epoch-" + str(num_epoch),
            suffix=".tar",
        )
    torch.save(
        {
            "epoch": num_epoch,
            "cv": num_cv,
            "model_state_dict": model_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )


def save_model(
    result_save_dir: str,
    dataset_name: str,
    num_cv: int,
    num_epoch: int,
    model_net: Type[torch.nn.Module],
) -> None:
    """
    The save_model function saves the model to a file.

    The save_model function saves the model to a file. The filename is constructed from the dataset name, number of cross-validation folds, and number of epochs trained on.

    :param result_save_dir:str: Used to specify the directory where the model will be saved.
    :param dataset_name:str: Used to name the file.
    :param num_cv:int: Used to make the filename unique.
    :param num_epoch: Used to save the model after a certain number of epochs.
    :param model_net:Type[torch.nn.Module]: Used to save the model.
    :param : Used to specify the directory where the model will be saved.
    :return: The path of the saved model.

    :doc-author: Julian M. Kleber
    """
    prefix = result_save_dir + "/checkpoints/CV_" + str(num_cv)
    check_make_dir(prefix)
    save_path = prepare_file_name_saving(
        prefix=prefix,
        file_name=dataset_name + "_Epoch-" + str(num_epoch),
        suffix=".pt",
    )
    torch.save(model_net.state_dict(), save_path)


def load_model_parameters(model_params_file_path: str) -> Dict[Any, Any]:
    """
    The load_model_parameters function loads the model parameters from a JSON file.

    Parameters:

        model_params_file_path (str): The path to the JSON file containing the model parameters.

    Returns:

        dict: A dictionary of all of the loaded model parameters.

    :param model_params_file_path:str: Used to Specify the file path of the model parameters.
    :return: A dictionary of model parameters.

    :doc-author: Julian M. Kleber
    """

    return load_json_from_file(model_params_file_path)


def save_model_parameters(model_net: Model, save_dir: str) -> None:
    """
    The save_model_parameters function saves the model architecture to a csv file.

        Args:
            model_net (torch.nn.Module): The neural network that is being used for training and testing, e.g., CNN() or RNN().
            save_path (str): The path where the json file will be saved to, e.g., "./model/".

        Returns: None

    :param model_net:Type[torch.nn.Module]: Used to specify the type of model that is being used.
    :param save_path:str: Used to save the model architecture in a json file.
    :return: A dictionary of the model architecture (model_architecture).

    :doc-author: Julian M. Kleber
    """
    prefix = save_dir + "/model_parameters/"
    model_architecture = model_net.__dict__
    file_name = prepare_file_name_saving(
        prefix=prefix, file_name="model_architecture", suffix=".json"
    )
    save_json_to_file(model_architecture, file_name=file_name)


def get_latest_checkpoint(search_dir: str, num_cv: int, epoch: int) -> str:

    if not search_dir.endswith("/"):
        search_dir += "/"

    search_dir += "checkpoints"
    checkpoint_dirs = os.listdir(search_dir)
    correct_sub_dir = checkpoint_dirs[checkpoint_dirs.index(
        "CV_" + str(num_cv))]
    search_dir += "/" + correct_sub_dir
    checkpoints = os.listdir(search_dir)
    checkpoints = sorted(checkpoints)
    return checkpoints[:-1]
