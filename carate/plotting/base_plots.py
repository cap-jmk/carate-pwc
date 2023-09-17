"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""
from typing import Type, Optional, Dict, Any, List, Tuple
import logging
import numpy as np
import matplotlib.pyplot as plt
from amarium.utils import load_json_from_file, prepare_file_name_saving, attach_slash

from carate.statistics.analysis import (
    get_avg,
    get_min,
    get_max,
    unpack_run,
    get_stacked_list,
    load_result_json,
    get_min_max_avg_cv_run,
)


def plot_range_band_multi(
    result: List[Dict[str, float]],
    key_vals: List[str],
    file_name: str,
    alpha: float = 0.5,
    save_dir: Optional[str] = None,
    title_text: Optional[str] = None,
) -> None:
    """
    The plot_range_band_multi function is used to plot multiple range bands on the same graph.
    The function takes in a list of dictionaries, each dictionary containing the results from one
    cross-validation run.It also takes in a list of keys that correspond to values within each
    dictionary that should be plotted as range bands. The function then plots all these values as
    separate lines with their corresponding ranges filled in between them.

    :param result:List[Dict[str: Used to Pass in the list of dictionaries
    :param float]]: Used to Set the alpha value of the fill.
    :param key_vals:List[str]: Used to Specify which metrics to plot.
    :param file_name:str: Used to name the file that will be saved.
    :param alpha:float=0.5: Used to Set the transparency of the fill between.
    :param save_dir:Optional[str]=None: Used to Save the plot in a specific directory.
    :param : Used to Set the transparency of the fill between min and max values.
    :return: A plot of the average, maximum and minimum values.

    :doc-author: Julian M. Kleber
    """

    fig, axis = plt.subplots()
    axis.set_xlabel("Training step")

    for i in range(len(key_vals)):
        max_val: List[float]
        min_val: List[float]
        avg_val: List[float]

        max_val, min_val, avg_val = get_min_max_avg_cv_run(
            result=result, key_val=key_vals[i]
        )
        logging.info(f"Max values are: {max_val}")
        logging.info(f"Min values are: {min_val}")
        logging.info(f"Avg values are: {avg_val}")

        axis.plot(avg_val, label=key_vals[i])
        plot_range_fill(max_val, min_val, alpha, axis)

    axis.set_ylabel("Value")
    axis.legend()
    axis.set_title(title_text)
    save_publication_graphic(fig_object=fig, file_name=file_name, prefix=save_dir)


def plot_range_band_single(
    result: List[Dict[str, List[float]]],
    key_val: str,
    file_name: str,
    alpha: float = 0.5,
    fixed_y_lim=(0.0, 1.01),
    save_dir: Optional[str] = None,
    legend_text: Optional[str] = None,
) -> None:
    """
    The plot_range_band function takes in a list of dictionaries, each dictionary containing the
    results from one run.
    It then plots the average value for each key_val (e.g., 'accuracy') and also plots a
    range band between the minimum and maximum values for that key_val across all runs.

    :param result:List[Dict[str: Used to plot the results of each run.
    :param float]]: Used to specify the type of data that is being passed into the function.
    :param key_val:str: Used to specify which key in the dictionary to plot.
    :param file_name:str: Used to save the plot as a png file.
    :return: A plot with the average value of a list, and the minimum and maximum values.

    :doc-author: Julian M. Kleber
    """
    max_val: List[float]
    min_val: List[float]
    avg_val: List[float]

    max_val, min_val, avg_val = get_min_max_avg_cv_run(result=result, key_val=key_val)

    fig, axis = plt.subplots()
    if legend_text is not None:
        axis.plot(avg_val, "-", label=legend_text)
    else:
        axis.plot(avg_val, "-", label=legend_text)
    plot_range_fill(max_val, min_val, alpha, axis)

    axis.set_ylim(*fixed_y_lim)
    axis.set_ylabel(key_val)
    if legend_text is not None:
        axis.legend()

    axis.set_xlabel("Training step")

    save_publication_graphic(fig_object=fig, file_name=file_name, prefix=save_dir)


def plot_range_fill(
    max_val: List[float], min_val: List[float], alpha: float, axis
) -> None:
    """
    The plot_range_lines function takes in three lists of floats, max_val, min_val and avg_val.
    It then plots the average value as a line graph with the training steps on the x-axis and
    the average values on the y-axis. It also fills in between each point with a color to show
    the range of values for that particular step.

    :param max_val:List[float]: Used to Plot the maximum value of each training step.
    :param min_val:List[float]: Used to Plot the minimum value of each metric.
    :param avg_val:List[float]: Used to Plot the average value of a given metric.
    :return: A plot with the average value, max value and min values for each training step.

    :doc-author: Julian M. Kleber
    """

    training_steps = np.arange(0, len(max_val), 1)
    axis.fill_between(training_steps, min_val, max_val, alpha=alpha)


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


def parse_old_file_format_for_plot(
    path_to_json: str,
) -> List[List[float]]:  # pragma no-cover
    """
    The parse_old_file_format_for_plot function takes in a path to a json file and returns the
    following:

    :param path_to_json:str: Used to Specify the path to the json file that we want to parse.
    :return: A list of dictionaries.

    :doc-author: Julian M. Kleber
    """

    result_acc = load_result_json(path_to_json=path_to_json)
    train_frames_acc = parse_min_max_avg(result_acc["Train_acc"])
    test_frames_acc = parse_min_max_avg(result_acc["Test_acc"])
    return train_frames_acc, test_frames_acc


def load_result_json_old_format(path_to_json: str) -> Dict[str, Any]:
    """
    The load_result_json function takes in a path to a json file and returns the contents of
    that json file as a dictionary.The function also parses the "Acc" key in the dictionary, which
    contains lists of tuples containing train and test accuracy values.
    The function then separates these tuples into two separate lists, one for train accuracy values
    and one for test accuracy values.These new lists are added to the original dictionary under
    keys "Acc_train" and "Acc_val", respectively.

    :param path_to_json:str: Used to Specify the path to the json file.
    :return: A dictionary with the following keys:.

    :doc-author: Julian M. Kleber
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
    result["Acc_train"] = train_acc
    result["Acc_val"] = test_acc

    return result


def parse_min_max_avg(result_list: List[List[float]]) -> List[float]:
    """
    The parse_min function takes a list of lists and returns the minimum value for each sublist.

    :param result_list:List[List[float]]: Used to Store the results of the simulation.
    :return: A list of the minimum values for each step.

    :doc-author: Julian M. Kleber
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
