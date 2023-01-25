"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""
from typing import Type, Optional, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_range_band(result:List[Dict[str, Any]])->None:
    """
    The plot_range_band function plots a range band.
    
    The plot_range_band function plots a range band. The plot is created using the matplotlib library and the figure is returned as an object that can be used to further customize the plot before displaying it or saving it to file.
    
    :param result:List[Dict[str, Any]]: Used to Plot the data in a list of dictionaries.
    :param: Used to Specify the type of data that is being passed into the function.
    :return: A plot of the y values and their error.
    
    :doc-author: Julian M. Kleber
    """

def get_stacked_list(path_to_directory:str, column_name:str): 
    results = []
    for i in range(5):
        result = parse_acc_list_json(
            path_to_json=f"/home/dev/carate/carate/carate_config_files/Classification/PROTEINS_20/data/CV_0/PROTEINS_Epoch_4980.json"
        )
        results.append(result) # TODO use a collections advanced container. the one that stores conatiners, forgot the name 
    results = np.array(results)



def parse_old_file_format_for_plot(stacked_list:List[float])->None: 
    """
    The parse_old_file_format_for_plot function takes in a path to a json file and returns the following:
    
    :param path_to_json:str: Used to Specify the path to the json file that we want to parse.
    :return: A list of dictionaries.
    
    :doc-author: Julian M. Kleber
    """
    

    result_acc = parse_acc_list_json(path_to_json=path_to_json)
    train_frames_acc = parse_min_max_avg(result_acc["Train_acc"])
    test_frames_acc = parse_min_max_avg(result_acc["Test_acc"])
    print(train_frames_acc)
    print(test_frames_acc)


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
    

    result = load_json_from_file(
        path_to_json
    )
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

def parse_min_max_avg(result_list:List[List[float]])->List[float]:
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

        step_list  = result_list[:, i]
        minimum = get_min(step_list=step_list)
        maximum = get_max(step_list=step_list)
        average = get_avg(step_list=step_list)
        minima.append(minimum)
        maxima.append(maximum)
        averages.append(average)
    return minima, maxima, averages

def get_avg(step_list:List[float])->float: 
    """
    The get_avg function takes a list of floats and returns the average value.
        
    
    :param step_list:List[float]: Used to Tell the function that it will be taking a list of floats as an argument.
    :return: The mean of the step_list.
    
    :doc-author: Julian M. Kleber
    """
    
    return np.mean(step_list)

def get_max(step_list:List[float])->float: 
    """
    The get_max function takes a list of floats and returns the maximum value in that list.

    
    :param step_list:List[float]: Used to Tell the function that step_list is a list of floats.
    :return: The maximum value in the list.
    
    :doc-author: Trelent
    """
    

    return np.max(step_list)


def get_min(step_list:List[float])->float: 
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


