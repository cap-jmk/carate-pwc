"""
Module for handling config files used to start 
the training and evaluation process 

:author: Julian M. Kleber
"""

from typing import Optional, List, Dict, Any

import subprocess
from amarium.utils import save_json_to_file


def convert_py_to_json(
    file_name: str, out_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    The convert_py_to_json function takes in a file name and an output file name.
    It reads the input file, which is assumed to be a .py config file with key value pairs separated by spaces.
    The function then converts the key value pairs into a dictionary and saves it as json format in the output filename.

    :param file_name:str: Used to Specify the file name of the.
    :param out_name:str: Used to Specify the name of the output file.
    :return: A dictionary with the same information as the file.

    :doc-author: Julian M. Kleber
    """

    json_dict: Dict[str, Any] = {}
    lines = read_file(file_name)

    for line in lines:
        tmp = line.split(" ")
        if tmp[2] == "None" or tmp[2] == "None":
            json_dict[tmp[0]] = None
        else:
            json_dict[tmp[0]] = tmp[2]

    if out_name is not None:
        save_json_to_file(json_dict, file_name=out_name)

    return json_dict


def read_file(file_name: str) -> List[str]:
    """
    The read_file function takes a file name as input and returns the contents of that file as a list of strings.
    The function also formats the code using black before reading it in.

    :param file_name:str: Used to Specify the file name.
    :return: A list of strings.

    :doc-author: Julian M. Kleber
    """

    subprocess.run(
        ["black", file_name], capture_output=True
    )  # format file to avoid misunderstandings
    with open(file_name, "r", encoding="utf-8") as file:
        raw = file.readlines()
    result = sanitize_raw_py(raw)
    return result


def sanitize_raw_py(raw_input: List[str]) -> List[str]:
    """
    The sanitize_raw_py function takes in a list of strings and returns a list of strings.
    The function removes all newline characters from the input, as well as any quotation marks or apostrophes.

    :param raw_input:list[str]: Used to Specify the type of data that is expected to be passed into the function.
    :return: A list of strings.

    :doc-author: Julian M. Kleber
    """

    result = []
    for line in raw_input:
        if line == "\n":
            continue
        else:
            result.append(line.replace("\n", "").replace(
                '"', "").replace("'", ""))
    return result
