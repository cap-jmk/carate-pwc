"""
Perform tests on the functionality of running models against
a regression dataset

:author: Julian M. Kleber
"""
import os
import shutil
import torch

from typing import Type
import logging

from amarium.utils import search_subdirs

from carate.run import RunInitializer
import carate.models.cgc_regression as CGCR
from carate.evaluation.regression import RegressionEvaluation
from carate.load_data import StandardDatasetMoleculeNet, StandardDatasetTUDataset


logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)



def test_regression_multitaksing():

    check_dir_paths()
    config_filepath = "tests/config/regression_alchemy_test.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = "tests/results/ALCHEMY_test"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function
    
    check_result_files(result_dir=result_dir)


def test_regression_override():

    check_dir_paths()
    config_filepath = "tests/config/regression_test_config_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = "tests/results/ZINC_test"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    check_result_files(result_dir=result_dir)
    
def test_regression_no_override():

    check_dir_paths()

    config_filepath = "tests/config/regression_test_config_no_override.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    result_dir = "tests/results/ZINC_test/"
    assert str(runner.data_set) == "StandardTUDataset"
    runner.run()  # takes instance attributes as parameters for the run() function

    check_result_files(result_dir=result_dir, override=True)

def check_result_files(result_dir:str, override:bool=False)->bool: 
    """
    The check_files function checks the files in the result directory.
    It checks if there are two subdirectories called CV_0 and CV_2, which contain a file called 
    ZINC_test.json.Furthermore it checks if there is a subdirectory called checkpoints, which 
    contains two directories named CV_0 and CV_2 with a file named ZINC-test.tar inside each of 
    them.
    
    :param result_dir:str: Used to Specify the directory where the results are stored.
    :return: True if all the files are found.
    
    :doc-author: Trelent
    """
    


    if not result_dir.endswith("/"): 
        result_dir += "/"


    #base dir to check the files in    
    data_dir = result_dir+"data/"
    checkpoint_dir = result_dir+"checkpoints/"
    model_parameter_dir = result_dir +"model_parameters/"

    # check result files
    result_files, result_dirs = search_subdirs(dir_name=data_dir)
    reference_dirs = [
        data_dir+"CV_0",
        data_dir+"CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        data_dir+"CV_0/ZINC_test.json",
        data_dir+"CV_1/ZINC_test.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name=checkpoint_dir
    )

    reference_dirs = [
        checkpoint_dir + "CV_0",
        checkpoint_dir + "CV_1",
    ]

    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    if override == False:
        assert len(result_files) == 2
        reference_files = [
            checkpoint_dir+"CV_0/ZINC_test.tar",
            checkpoint_dir+"CV_1/ZINC_test.tar",
        ]
       
    else:
        assert len(result_files) == 4
        reference_files = [
            checkpoint_dir+"CV_0/ZINC_test_Epoch-1.tar",
            checkpoint_dir+"CV_0/ZINC_test_Epoch-2.tar",
            checkpoint_dir+"CV_1/ZINC_test_Epoch-1.tar",
            checkpoint_dir+"CV_1/ZINC_test_Epoch-2.tar",
        ] 
    for name in result_files:
                assert name in reference_files
    result_dir_content = os.listdir(model_parameter_dir)
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data

    return True


def check_dir_paths():
    """
    The check_dir_paths function deletes the data and results directories if they exist.
    This is to ensure that the tests are run on a clean slate.
    
    :return: None
    
    :doc-author: Julian M. Kleber
    """
    
    if os.path.isdir("tests/data"):
        shutil.rmtree("tests/data")
    if os.path.isdir("tests/results"):
        shutil.rmtree("tests/results")
