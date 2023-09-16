import os, shutil

from amarium.utils import search_subdirs


def check_result_files(
    result_dir: str, run_title: str, data_set_name: str, override: bool = False
) -> bool:
    """
    The check_result_files function checks the result files in the directory specified by result_dir.
    The function checks if there are two subdirectories, CV_0 and CV_2, which contain a
    file ZINC_test.json each. It also checks if there are two tar files in the checkpoint directory
    with names ZINC_test.tar and ZINC-CV-0/ZINC-CV-2 respectively.

    :param result_dir:str: Used to Specify the directory where the results are saved.
    :param run_title:str: Used to Create a unique directory name for the results.
    :param override:bool=False: Used to Check if the override parameter is set to false, which means
     that it will only save the best model.
    :return: True if the result files are present and false otherwise.

    :doc-author: Trelent
    """

    if not result_dir.endswith("/"):
        result_dir += "/"

    # base dir to check the files in
    data_dir = result_dir + "data/"
    checkpoint_dir = result_dir + "checkpoints/"
    model_parameter_dir = result_dir + "model_parameters/"

    # check result files
    result_files, result_dirs = search_subdirs(dir_name=data_dir)
    reference_dirs = [
        data_dir + "CV_0",
        data_dir + "CV_1",
    ]
    assert len(result_dirs) == 2, str(len(result_dirs))
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        data_dir + f"CV_0/{data_set_name}.json",
        data_dir + f"CV_1/{data_set_name}.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(dir_name=checkpoint_dir)

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
            checkpoint_dir + f"CV_0/{run_title}.tar",
            checkpoint_dir + f"CV_1/{run_title}.tar",
        ]

    else:
        assert len(result_files) == 4
        reference_files = [
            checkpoint_dir + f"CV_0/{data_set_name}_Epoch-1.tar",
            checkpoint_dir + f"CV_0/{data_set_name}_Epoch-2.tar",
            checkpoint_dir + f"CV_1/{data_set_name}_Epoch-1.tar",
            checkpoint_dir + f"CV_1/{data_set_name}_Epoch-2.tar",
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


def check_plotting_dir() -> None:
    """
    The check_plotting_dir function checks to see if the plots directory exists. If it does, it
    deletes the directory and all of its contents.

    :return: None

    :doc-author: Trelent
    """

    if os.path.isdir("./plots"):
        shutil.rmtree("./plots")
