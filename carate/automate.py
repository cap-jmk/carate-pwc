"""Module providing a CLI 
interface for training models#

:author: Julian  M. Kleber
"""
from typing import Type
import click

from carate.run import RunInitializer, Run


@click.command()
@click.option("-c", help="Path to config file")
def train_algorithm(c: str) -> None:
    """
    The train_algorithm function takes in a config file and an output directory.
    It then runs the algorithm using the configuration specified in the config file,
    and saves all of its outputs to the output directory.

    :param c:str: Used to Specify the path to the configuration file.
    :param o:str: Used to Specify the output directory.
    :return: None.

    :doc-author: Julian M. Kleber
    """
    if c is None:
        raise RuntimeError("Please provide the path to a config file.")
    config_filepath = c
    runner = RunInitializer.from_file(config_filepath=config_filepath)

    runner.run()


if __name__ == "__main__":
    train_algorithm()
