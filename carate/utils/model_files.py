"""
Utility file for model checkpoints file operations

:author: Julian M. Kleber
"""

import torch

from carate.utils.file_utils import (
    prepare_file_name_saving,
    load_json_from_file,
    make_full_filename,
    save_json_to_file,
)


def load_model(
    model_path: str, model_params_path: str, model_net: type(torch.nn.Module)
):
    """
    The load_model function takes in a model_path, model_params_path and the type of network to be loaded.
    It then loads the parameters from the params file into a dictionary and uses that to create an instance of
    the specified network. It then loads in the state dict from PATH and sets it as eval mode.

    :param model_path:str: Used to specify the path to the model file.
    :param model_params_path:str: Used to load the model parameters from a file.
    :param model_net:type(torch.nn.Module): Used to specify the type of model that is being loaded.
    :return: A model that is loaded with the parameters in the path.

    :doc-author: Julian M. Kleber
    """

    parameters = load_model_parameters(model_params_path)
    model = model_net(**parameters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(
    result_save_dir: str,
    dataset_name: str,
    num_cv: int,
    num_epoch,
    model_net: type(torch.nn.Module),
):
    """
    The save_model function saves the model to a file.

    The save_model function saves the model to a file. The filename is constructed from the dataset name, number of cross-validation folds, and number of epochs trained on.

    :param result_save_dir:str: Used to specify the directory where the model will be saved.
    :param dataset_name:str: Used to name the file.
    :param num_cv:int: Used to make the filename unique.
    :param num_epoch: Used to save the model after a certain number of epochs.
    :param model_net:type(torch.nn.Module): Used to save the model.
    :param : Used to specify the directory where the model will be saved.
    :return: The path of the saved model.

    :doc-author: Julian M. Kleber
    """
    prefix = result_save_dir + "/checkpoints/"
    save_path = prepare_file_name_saving(
        prefix=prefix,
        file_name=dataset_name + "_CV-" +
        str(num_cv) + "_Epoch-" + str(num_epoch),
        suffix=".pt",
    )
    torch.save(model_net.state_dict(), save_path)


def load_model_parameters(model_params_file_path: str) -> dict:
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


def save_model_parameters(model_net: type(torch.nn.Module), save_dir: str) -> None:
    """
    The save_model_parameters function saves the model architecture to a csv file.

        Args:
            model_net (torch.nn.Module): The neural network that is being used for training and testing, e.g., CNN() or RNN().
            save_path (str): The path where the json file will be saved to, e.g., "./model/".

        Returns: None

    :param model_net:type(torch.nn.Module): Used to specify the type of model that is being used.
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
