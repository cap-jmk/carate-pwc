"""Module for routine multi-run plotting.

:author: Julian M. Kleber
"""
import os
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt

from amarium.utils import attach_slash


from carate.statistics.analysis import (
    get_min_max_avg_cv_run,
    get_stacked_list,
    get_max_average,
    get_min_average
)
from carate.plotting.base_plots import plot_range_fill, save_publication_graphic

import logging

logger = logging.getLogger(__name__)


def plot_all_runs_in_dir(
    base_dir: str,
    save_name: str,
    val_single: str = "Acc_test",
    val_multi: Tuple[str] = ("Acc_train", "Acc_test"),
    y_lims=(0.0, 1.01),
) -> None:

    run_dirs = os.listdir(base_dir)
    fig, axis = plt.subplots()
    for run_dir in run_dirs:
        full_dir = attach_slash(base_dir) + attach_slash(run_dir) + attach_slash("data")
  
            # encapsulate into function

        name = os.listdir(full_dir + attach_slash("CV_0"))[0]

        logger.info("Full dir for run to plot:", full_dir)
        legend_text = full_dir.split("/")[-3]
        logger.info("Plotting: ", legend_text)
        result = get_stacked_list(
            path_to_directory=full_dir,
            num_cv=5,
            json_name=name,
        )
        mean, std = get_max_average(result, val_single)
        logger.info(full_dir, " - Mean : ", mean, " Std: ", std)
        plot_range_band_multi_run(
            result,
            fixed_y_lim=y_lims,
            key_val=val_single,
            file_name=f"{legend_text}_{val_single}",
            save_dir="./plots",
            alpha=0.4,
            legend_text=legend_text,
            fig=fig,
            axis=axis
        )
        
    save_publication_graphic(fig_object=fig, file_name=save_name)


def plot_range_band_multi_run(
    result: List[Dict[str, List[float]]],
    key_val: str,
    file_name: str,
    fig,
    axis,
    alpha: float = 0.5,
    fixed_y_lim=(0.0, 1.01),
    save_dir: Optional[str] = None,
    legend_text: Optional[str] = None,
) -> None:
    """The plot_range_band function takes in a list of dictionaries, each
    dictionary containing the results from one run. The function is meant to be
    used in a for-loop iterating about many runs. It then plots the average
    value for each key_val (e.g., 'accuracy') and also plots a range band
    between the minimum and maximum values for that key_val across all runs.

    :param result: List[Dict[str: Used to plot the results of each run.
        :param float]]: Used to specify the type of data that is being
        passed into the function.
    :param key_val: str: Used to specify which key in the dictionary to
        plot.
    :param file_name: str: Used to save the plot as a png file.
    :return: A plot with the average value of a list, and the minimum
        and maximum values. :doc-author: Julian M. Kleber
    """
    max_val: List[float]
    min_val: List[float]
    avg_val: List[float]

    max_val, min_val, avg_val = get_min_max_avg_cv_run(result=result, key_val=key_val)

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
