import os
import json
import time
import logging

LOGGER = logging.getLogger(__name__)


timestr = time.strftime("%Y%m%d-%H%M%S")


def prepare_file_name_saving(prefix: str, file_name: str, ending: str) -> str:
    """
    The prepare_file_name_saving function takes a prefix and file name as input. It checks to see if the directory exists, and makes it if not. Then it returns the full path of the file.

    :param prefix:str: Used to Specify the folder where the file will be saved.
    :param file_name:str: Used to Specify the name of the file to be saved.
    :return: The full path of the file name.

    :doc-author: Trelent
    """
    check_make_dir(prefix)
    file_name = make_full_filename(prefix, file_name)
    file_name = check_file_name(file_name, ending)
    return file_name


def insert_string_in_file_name(file_name: str, insertion: str, ending: str):

    root, ext = os.path.splitext(file_name)

    if ending is None and not ext:
        raise ValueError(
            "You must either specify an ending in the file_name or pass an ending through the ending argument. For example the file_name could be 'foo.bar' or you pass file_name'foo' with ending = '.bar'"
        )
    if ending is not None:
        if not ending.startswith("."):
            ending = "." + ending

    if not ext:
        file_name = file_name + "_" + insertion + ending
    else:
        file_name = root + "_" + insertion + ext
    return file_name


def prepare_data_decicion_lib(data_set: object, columns: list = None) -> tuple():
    """
    The prepare_data_decicion_lib function takes in a data set from scikitlearn and returns the X and y values for the decision boundary plot.
    The function can also take in a list of columns to use as features, but if no input is given then it will default to using all
    the features.

    :param data_set:object: Used to Pass the data set to the function.
    :param columns:list=None: Used to Select the columns of the data set that are used for training.
    :return: A tuple containing the x and y values.

    :doc-author: Trelent
    """

    if columns is None:
        X = data_set.data[:, :2]
    else:
        assert (
            len(columns) == 2
        ), "Length of the columns input must be equalt to two. Otherwise the plotting of the decision boundary can't work"
        X = np.zeros((len(data_set.target), 2))
        X[:, 0] = data_set.data[:, columns[0]]
        X[:, 1] = data_set.data[:, columns[1]]
    y = data_set.target
    return (X, y)


def check_file_name(file_name: str, ending: str):
    """
    The check_name_plot function checks that the file_name ends with .png.
    If it does not, then check_name_plot appends .png to the end of the file_name.

    :param file_name: Used to Specify the name of the file to be plotted.
    :return: The file_name with the correct file extension.

    :doc-author: Trelent
    """

    root, ext = os.path.splitext(file_name)
    if not ext:
        if not ending.startswith("."):
            ending = "." + ending
        file_name += ending
    return file_name


def make_full_filename(prefix, file_name):
    """
    The make_full_filename function takes a prefix and a file_name as input.
    If the prefix is None, then the file_name is returned unchanged.
    Otherwise, if the file name starts with 'http://' or 'ftp://', then it's assumed to be an URL and
    the full_filename will contain both the prefix and file_name; otherwise, only return full_filename = file_name.

    :param prefix: Used to Add a prefix to the file_name.
    :param file_name: Used to Create a full file_name for the file to be downloaded.
    :return: The full file_name with the prefix added to the beginning of the file_name.

    :doc-author: Trelent
    """
    if prefix is None:
        file_name = file_name

    if prefix.endswith("/") and file_name.startswith("/"):
        file_name = prefix + file_name[1:]
    elif prefix.endswith("/") or file_name.startswith("/"):
        file_name = prefix + file_name
    else:
        file_name = prefix + "/" + file_name
    return file_name


def check_make_dir(dir_name: str) -> None:
    """
    The check_make_dir function checks if a directory exists. If it does not exist, the function creates it.

    :param dir_name:str: Used to Specify the folder name.
    :return: None.

    :doc-author: Trelent
    """

    # You should change 'test' to your preferred folder.
    check_folder = os.path.isdir(dir_name)
    LOGGER.info("Checked the directory " + str(dir_name))
    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(dir_name)
        LOGGER.info("created folder : " + str(dir_name))

    else:
        LOGGER.info(dir_name + "folder already exists.")


def load_json_from_file(file_name: str):
    import json

    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def save_json_to_file(dictionary, file_name: str = None, suffix: str = ".json"):
    """
    The save_json function saves a dictionary to a json file.

    :param dictionary: Used to store the data that will be saved.
    :param file_name:str=None: Used to specify a file name.
    :return: A string with the name of the file that was just created.

    :doc-author: Julian M. Kleber
    """
    import numpy as np

    def convert(o):

        if isinstance(o, np.generic):
            return o.item()
        elif isinstance(o, np.ndarray):
            return list(o)

    if file_name is None:

        file_name = make_date_file_name(suffix=suffix)
    with open(file_name, "w") as out_file:
        json.dump(dictionary, out_file, default=convert)
        LOGGER.info("Saved json " + str(file_name))


def make_date_file_name(prefix: str = "", file_name: str = "", suffix: str = ""):
    """
    The make_date_file_name function creates a file name with the date and time
    stamp in the format YYYY-MM-DD_HH:MM:SS.out.  The prefix, file_name, and suffix
    are optional arguments that can be used to specify what string should precede
    the date stamp in the file_name (prefix), what string should be appended after
    the date stamp (suffix), or both (file_name).   If no arguments are provided,
    then make_date_file_name will use default values for all three arguments.

    :param prefix:str="": Used to Add a prefix to the file name.
    :param file_name:str="": Used to Specify the name of the file.
    :param suffix:str=".out": Used to Specify the file extension.
    :return: A string that is the combination of prefix, file_name and suffix.

    :doc-author: Trelent
    """
    # TODO: should not have prefix and suffix in it
    timestr = time.strftime("%Y%m%d-%H%M%S" + prefix + file_name + suffix)
    return timestr


def get_grid_positions(rows: int, cols: int):

    grid = []
    for i in range(2, rows):
        for j in range(cols):
            grid.append((i, j))
    return grid
