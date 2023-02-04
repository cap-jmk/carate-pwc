"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""
from typing import Type, Optional, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import ChainMap


def plot_range_band(result: List[Dict[str, float]], key_val: str) -> None:

    arr_res = np.zeros((len(result), len(result[0][key_val])))
    print(arr_res.shape)
    for i, res in enumerate(result):
        print((len(res[key_val])))
        arr_res[i, :] = res[key_val] #TOOD fix the shapes


def get_stacked_list(
    path_to_directory: str, column_name: str, num_cv: int, json_name: str
) -> List[Dict[str, float]]:

    results = []

    for i in range(num_cv):
        try:
            result = parse_acc_list_json(
                path_to_json=path_to_directory + f"CV_{i}/" + json_name
            )
            results.append(result)

        except Exception as exc:
            print(exc)
            continue

    return results


def parse_old_file_format_for_plot(stacked_list: List[float]) -> List[List[float]]:
    """
    The parse_old_file_format_for_plot function takes in a path to a json file and returns the following:

    :param path_to_json:str: Used to Specify the path to the json file that we want to parse.
    :return: A list of dictionaries.

    :doc-author: Julian M. Kleber
    """

    result_acc = parse_acc_list_json(path_to_json=path_to_json)
    train_frames_acc = parse_min_max_avg(result_acc["Train_acc"])
    test_frames_acc = parse_min_max_avg(result_acc["Test_acc"])
    return train_frames_acc, test_frames_acc


def parse_acc_list_json(path_to_json: str) -> Dict[str, Any]:
    """
    The parse_acc_list_json function takes in a path to a json file and returns the contents of that json file as a dictionary.
    The function also parses the "Acc" key in the dictionary, which contains lists of tuples containing train and test accuracy values.
    The function then separates these tuples into two separate lists, one for train accuracy values and one for test accuracy values.
    These new lists are added to the original dictionary under keys "Acc_train" and "Acc_val", respectively.

    :param path_to_json:str: Used to Specify the path to the json file.
    :return: A dictionary with the following keys:.

    :doc-author: Trelent
    """

    result = load_json_from_file(path_to_json)
    acc_store = result["Acc"]

    train_acc = []
    test_acc = []

    for i in range(len(acc_store)):
        train, test = acc_store[i]
        train_acc.append(train)
        test_acc.append(test)

    del result["Acc"]
    print(f"Len of train, {len(train_acc)}")
    result["Acc_train"] = train_acc
    result["Acc_val"] = test_acc

    return result


def parse_min_max_avg(result_list: List[List[float]]) -> List[float]:
    """
    The parse_min function takes a list of lists and returns the minimum value for each sublist.

    :param result_list:List[List[float]]: Used to Store the results of the simulation.
    :return: A list of the minimum values for each step.

    :doc-author: Trelent
    """

    minima = []
    maxima = []
    averages = []
    for i in range(len(result_list[0])):

        step_list = result_list[:, i]
        minimum = get_min(step_list=step_list)
        maximum = get_max(step_list=step_list)
        average = get_avg(step_list=step_list)
        minima.append(minimum)
        maxima.append(maximum)
        averages.append(average)
    return [minima, maxima, averages]


def get_avg(step_list: List[float]) -> float:
    """
    The get_avg function takes a list of floats and returns the average value.


    :param step_list:List[float]: Used to Tell the function that it will be taking a list of floats as an argument.
    :return: The mean of the step_list.

    :doc-author: Julian M. Kleber
    """

    return np.mean(step_list)


def get_max(step_list: List[float]) -> float:
    """
    The get_max function takes a list of floats and returns the maximum value in that list.


    :param step_list:List[float]: Used to Tell the function that step_list is a list of floats.
    :return: The maximum value in the list.

    :doc-author: Trelent
    """

    return np.max(step_list)


def get_min(step_list: List[float]) -> float:
    """
    The get_min function takes a list of floats and returns the minimum value in that list.

    :param step_list:List[float]: Used to Specify the type of parameter that is being passed into the function.
    :return: The minimum value in the step_list.

    :doc-author: Trelent
    """

    return np.min(step_list)


if __name__ == "__main__":
    from amarium.utils import load_json_from_file
    import pandas as pd

    path_to_directory = (
        "/home/dev/carate/carate/carate_config_files/Classification/PROTEINS_10/data/"
    )
    result = get_stacked_list(
        path_to_directory=path_to_directory,
        column_name="Acc",
        num_cv=5,
        json_name="PROTEINS_Epoch_4980.json",
    )
    plot_range_band(result, key_val="Acc_train")
