"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""
"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""


from typing import Type, Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from amarium.utils import load_json_from_file, prepare_file_name_saving
from collections import ChainMap


def plot_range_band_multi(
    result: List[Dict[str, float]],
    key_vals: List[str],
    file_name: str,
    save_dir: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("Training step")

    for i in range(len(key_vals)):
        max_val: List[float]
        min_val: List[float]
        avg_val: List[float]

        max_val, min_val, avg_val = unpack_cross_validation(
            result=result, key_val=key_vals[i]
        )

        ax.plot(avg_val, "-", label=key_vals[i])
        ax.set_ylim(0, 1.1)
        ax.fill_between(min_val, max_val, alpha=0.2)

    ax.set_ylabel("Value")
    ax.legend()

    save_publication_graphic(fig_object=fig, file_name=file_name, prefix=save_dir)


def plot_range_band_single(
    result: List[Dict[str, float]],
    key_val: str,
    file_name: str,
    save_dir: Optional[str] = None,
) -> None:
    """
    The plot_range_band function takes in a list of dictionaries, each dictionary containing the
    results from one run.
    It then plots the average value for each key_val (e.g., 'accuracy') and also plots a
    range band between the minimum and maximum values for that key_val across all runs.

    :param result:List[Dict[str: Used to Plot the results of each run.
    :param float]]: Used to Specify the type of data that is being passed into the function.
    :param key_val:str: Used to Specify which key in the dictionary to plot.
    :param file_name:str: Used to Save the plot as a png file.
    :return: A plot with the average value of a list, and the minimum and maximum values.

    :doc-author: Julian M. kleber
    """
    max_val: List[float]
    min_val: List[float]
    avg_val: List[float]

    max_val, min_val, avg_val = unpack_cross_validation(result=result, key_val=key_val)

    fig, ax = plt.subplots()
    ax.plot(avg_val, "-")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(key_val)
    ax.set_xlabel("Training step")
    ax.fill_between(min_val, max_val, alpha=0.2)

    save_publication_graphic(fig_object=fig, file_name=file_name, prefix=save_dir)


def unpack_cross_validation(
    result: List[Dict[str, float]], key_val: str
) -> Tuple[List[float]]:
    """
    The unpack_cross_validation function takes in a list of dictionaries, and a key value.
    It then unpacks the values associated with that key into three lists: max_val, min_val, avg_val.
    These lists are returned as a tuple.

    :param result:List[Dict[str: Used to Store the result of each iteration.
    :param float]]: Used to Store the results of the cross validation.
    :param key_val:str: Used to Specify which key in the dictionary to use for the unpacking.
    :return: The max, min and average value of the key_val parameter.

    :doc-author: Trelent
    """

    max_val = []
    min_val = []
    avg_val = []

    arr_res = np.zeros((len(result), len(result[0][key_val])))
    print(arr_res.shape)

    for i, res in enumerate(result):
        arr_res[i, :] = res[key_val]

    for i in range(arr_res.shape[1]):
        max_val.append(get_min(arr_res[:, i].tolist()))
        min_val.append(get_max(arr_res[:, i].tolist()))
        avg_val.append(get_avg(arr_res[:, i].tolist()))

    return (max_val, min_val, avg_val)


def save_publication_graphic(
    fig_object: Type[plt.figure], file_name: str, prefix: Optional[str] = None
) -> None:
    """
    The save_publication_graphic function saves the current figure to a file.

    The save_publication_graphic function saves the current figure to a file, with
    a default resolution of 300 dpi. The function also tightens up the layout of
    the plot before saving it, so that there is no wasted space around it in its saved form.

    :param file_name:str: Used to Specify the name of the file to be saved.
    :param prefix:Optional[str]=None: Used to Specify the directory where the file is saved.
    :return: None.

    :doc-author: Julian M. Kleber
    """

    file_name = prepare_file_name_saving(
        file_name=file_name, prefix=prefix, suffix=".png"
    )
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def get_stacked_list(
    path_to_directory: str, column_name: str, num_cv: int, json_name: str
) -> List[Dict[str, float]]:
    results = []

    for i in range(num_cv):
        print(f"Attempting cv {i}")

        result = parse_acc_list_json(
            path_to_json=path_to_directory + f"CV_{i}/" + json_name
        )
        results.append(result)
        print(f"parsed results for CV {i}")

    return results


def parse_old_file_format_for_plot(
    stacked_list: List[float], path_to_json: str
) -> List[List[float]]:
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
    return result


def parse_acc_list_json_old_format(path_to_json: str) -> Dict[str, Any]:
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
    print(len(acc_store))
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

    return float(np.mean(step_list))


def get_max(step_list: List[float]) -> float:
    """
    The get_max function takes a list of floats and returns the maximum value in that list.


    :param step_list:List[float]: Used to Tell the function that step_list is a list of floats.
    :return: The maximum value in the list.

    :doc-author: Trelent
    """

    return float(np.max(step_list))


def get_min(step_list: List[float]) -> float:
    """
    The get_min function takes a list of floats and returns the minimum value in that list.

    :param step_list:List[float]: Used to Specify the type of parameter that is being passed into the function.
    :return: The minimum value in the step_list.

    :doc-author: Trelent
    """

    return float(np.min(step_list))
