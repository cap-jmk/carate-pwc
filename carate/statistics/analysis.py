"""
Module to perform analysis of runs. 

:author: Julian M. Kleber
"""
from typing import List, Dict, Any
import numpy as np
from amarium.utils import append_slash, load_json_from_file
import logging

logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def get_best_average(stpe_list: List[float]):

    best_vals = np.max(step_list, axis=1)
    


def get_avg(step_list: List[float]) -> float:
    """
    The get_avg function takes a list of floats and returns the average value.


    :param step_list:List[float]: Used to Tell the function that it will be taking a list of
    floats as an argument.
    :return: The mean of the step_list.

    :doc-author: Julian M. Kleber
    """

    return float(np.mean(step_list))


def get_max(step_list: List[float]) -> float:
    """
    The get_max function takes a list of floats and returns the maximum value in that list.


    :param step_list:List[float]: Used to Tell the function that step_list is a list of floats.
    :return: The maximum value in the list.

    :doc-author: Julian M. Kleber
    """

    return float(np.max(step_list))


def get_min(step_list: List[float]) -> float:
    """
    The get_min function takes a list of floats and returns the minimum value in that list.

    :param step_list:List[float]: Used to Specify the type of parameter that is being passed
    into the function.
    :return: The minimum value in the step_list.

    :doc-author: Julian M. Kleber
    """

    return float(np.min(step_list))


def unpack_run(result_list:List[Dict[str, float]])->List[List[float]]:
    """
    The unpack_run function takes a list of dictionaries and returns a list of lists.
    The input is the output from the run_experiment function, which is a list of dictionaries.
    Each dictionary contains two keys: 'params' and 'val'. The value associated with the key 
    'params' is another dictionary containing all parameters used in that particular experiment. 
    The value associated with the key 'val' is an array containing all values returned by 
    each call to f(x). 
    

    :param result_list:List[Dict[str: Used to Specify the type of the input parameter.
    :param float]]: Used to Specify the type of data that is expected to be returned by the function.
    :return: A list of lists, where each sublist is the result of a run.
    
    :doc-author: Trelent
    """ 

    arr_res = np.zeros((len(result_list), len(result_list[0][key_val])))

    for i, res in enumerate(result_list):
        assert len(res[key_val]) == arr_res.shape[1], str(len(res[key_val])) + str(
            arr_res.shape[1]
        )
        arr_res[i, :] = res[key_val]
    return arr_res.to_list()

def get_stacked_list(
    path_to_directory: str, num_cv: int, json_name: str
) -> List[Dict[str, float]]:
    """
    The get_stacked_list function takes in a path to a directory, the name of the column
    that we want to stack, and the number of cross-validation folds. It then returns a list
    of dictionaries that contain all of our stacked results.

    :param path_to_directory:str: Used to Specify the directory where the json files are stored.
    :param column_name:str: Used to Specify the column name of the dataframe in which we
    want to get.
    :param num_cv:int: Used to Specify the number of cross validation runs.
    :param json_name:str: Used to Specify the name of the json file that will be parsed.
    :return: A list of dictionaries, where each dictionary is the accuracy for a single cv.

    :doc-author: Julian M. Kleber
    """

    results = []

    path_to_directory = append_slash(path_to_directory)

    for i in range(num_cv):
        logging.info(f"Attempting cv {i}")

        result = load_result_json(
            path_to_json=path_to_directory + f"CV_{i}/" + json_name
        )
        results.append(result)
        logging.info(f"parsed results for CV {i}")

    return results

def load_result_json(path_to_json: str) -> Dict[str, Any]:
    """
    The load_result_json function takes in a path to a json file and returns the contents of
    that json file as a dictionary.The function also parses the "Acc" key in the dictionary,
    which contains lists of tuples containing train and test accuracy values.
    The function then separates these tuples into two separate lists, one for train accuracy values
    and one for test accuracy values. These new lists are added to the original dictionary under
    keys "Acc_train" and "Acc_val", respectively.

    :param path_to_json:str: Used to Specify the path to the json file.
    :return: A dictionary with the following keys:.

    :doc-author: Julian M. Kleber
    """

    result = load_json_from_file(path_to_json)
    return result
