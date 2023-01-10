"""Module providing a CLI 
interface for training models#

:author: Julian  M. Kleber
"""
from typing import Type
import click

from carate.run import RunInitializer


@click.command()
@click.option("-c", help="Path to config file")
def train_algorithm(c: str, o: str) -> None:
    """
    The train_algorithm function takes in a config file and an output directory.
    It then runs the algorithm using the configuration specified in the config file,
    and saves all of its outputs to the output directory.

    :param c:str: Used to Specify the path to the configuration file.
    :param o:str: Used to Specify the output directory.
    :return: None.

    :doc-author: Trelent
    """

    config_filepath = c
    runner: Run
    runner = RunInitializer.from_file(config_filepath=config_filepath)

    runner.run()


if __name__ == "__main__":
    train_algorithm()
