"""Module providing a CLI
interface for training models#

:author: Julian  M. Kleber
"""
import os
from typing import Type
import click

from carate.runner.run import RunInitializer, Run


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-c", help="Path to config file")
@click.option("-d", help="Path to directory")
def carate(ctx, c, d):
    if ctx.invoked_subcommand is None:
        start_run(c=c, d=d)
    else:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand}")


def start_run(c: str, d: str) -> None:
    """
    The train_algorithm function takes in a config file and an output directory.
    It then runs the algorithm using the configuration specified in the config file,
    and saves all of its outputs to the output directory.

    :param c:str: Used to specify the path to the configuration file.
    :param d:str: Used to specify the path to the configuration file.

    :return: None.

    :doc-author: Julian M. Kleber
    """
    if c is None and d is None:
        raise RuntimeError("Please provide the path to a config file.")
    elif c != None and d != None:
        raise RuntimeError("Please provide either a directory or a file path.")
    elif c != None and d == None:
        config_filepath = c
        runner = RunInitializer.from_file(config_filepath=config_filepath)

        runner.run()
    elif d != None and c == None:
        train_whole_directory(d)


def train_whole_directory(d: str) -> None:
    """
    The train_algorithm function takes in a config file and an output directory.
    It then runs the algorithm using the configuration specified in the config file,
    and saves all of its outputs to the output directory.

    :param c:str: Used to specify the directory to the configuration files.
    :param o:str: Used to specify the output directory.
    :return: None.

    :doc-author: Julian M. Kleber
    """

    if d is None:
        raise RuntimeError("Please provide the path to a directory of config files")

    directory_file_path = d
    config_files = [
        file
        for file in os.listdir(directory_file_path)
        if (file.endswith(".py") or file.endswith(".json") or file.endswith(".yml"))
    ]
    if len(config_files) == 0:
        raise RuntimeError(
            f"The directory {directory_file_path} does not have any config files"
        )

    for config_file in config_files:
        if config_file.endswith(".py"):
            runner = RunInitializer.from_file(config_filepath=config_file)
        elif config_file.endswith(".json"):
            runner = RunInitializer.from_json(config_filepath=config_file)
        else:
            continue

        runner.run()
