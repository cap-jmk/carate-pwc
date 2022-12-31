"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""

import pandas as pd
import seaborn as sns
import seaborn.timeseries


def plot_min_max_shadow(data_dir: str):

    seaborn.timeseries._plot_range_band = _plot_range_band
    cluster_overload = pd.read_csv("TSplot.csv", delim_whitespace=True)
    cluster_overload["subindex"] = cluster_overload.groupby(
        ["Cluster", "Week"]
    ).cumcount()
    g = sns.FacetGrid(
        cluster_overload, row="Cluster", sharey=False, hue="Cluster", aspect=3
    )
    g = g.map_dataframe(custom_plot, "Week", "Overload", "subindex")
    plt.show()


def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):

    upper = data.max(axis=0)
    lower = data.min(axis=0)
    ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_band(*args, **kwargs)


def custom_plot(*args, **kwargs):

    df = kwargs.pop("data")
    pivoted = df.pivot(index="subindex", columns="Week", values="Overload")
    ax = sns.tsplot(
        pivoted.values, err_style="range_band", n_boot=0, color=kwargs["color"]
    )
