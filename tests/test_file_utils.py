import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carate.utils.file_utils import (
    check_file_name,
    make_full_filename,
    insert_string_in_file_name,
    load_json_from_file,
    save_json_to_file,
    search_subdirs
)


def test_check_file_name():
    def make_name(file_name, ending, expectation):
        new_file_name = check_file_name(file_name, ending)
        assert new_file_name == expectation, (
            " old file_name "
            + str(file_name)
            + " ending"
            + str(ending)
            + " returned file_name "
            + str(new_file_name)
        )

    # case 1
    file_name = "foo"
    ending = "bar"
    expectation = "foo.bar"
    make_name(file_name, ending, expectation)
    # case 2
    file_name = "foo.bar"
    ending = None
    expectation = "foo.bar"
    make_name(file_name, ending, expectation)


def test_make_full_name():
    def make_name(prefix, file_name, expectation):
        new_file_name = make_full_filename(prefix=prefix, file_name=file_name)
        assert new_file_name == expectation, (
            str(prefix)
            + " old file_name "
            + str(file_name)
            + " returned file_name "
            + str(new_file_name)
        )

    # case 1
    prefix = "tests"
    file_name = "testfile.png"
    expectation = "tests/testfile.png"
    make_name(prefix, file_name, expectation)
    # case 2
    prefix = "tests/"
    file_name = "testfile.png"
    expectation = "tests/testfile.png"
    make_name(prefix, file_name, expectation)
    # case 3
    prefix = "tests/"
    file_name = "/testfile.png"
    expectation = "tests/testfile.png"
    make_name(prefix, file_name, expectation)
    # case 3
    prefix = "tests"
    file_name = "/testfile.png"
    expectation = "tests/testfile.png"
    make_name(prefix, file_name, expectation)
    # case 4
    prefix = "tests/data"
    file_name = "testfile.png"
    expectation = "tests/data/testfile.png"
    make_name(prefix, file_name, expectation)


def test_insert_string_in_file_name():
    def make_name(file_name, insertions, ending, expectation):

        new_file_name = insert_string_in_file_name(
            file_name=file_name, insertion=insertion, ending=ending
        )
        assert new_file_name == expectation, (
            str(file_name)
            + " old file_name "
            + str(insertion)
            + str(ending)
            + " returned file_name "
            + str(new_file_name)
        )

    # case 1
    file_name = "foo.bar"
    insertion = "fantastic"
    ending = None
    expectation = "foo_fantastic.bar"
    make_name(file_name, insertion, ending, expectation=expectation)


def test_check_directory():
    pass


def check_name_plot():
    pass


def test_save_json_to_file():
    """
    The test_save_json_to_file function tests the saving and loading functionalities regarding
    json objects of the package
    :return: None.

    :doc-author: Julian M. Kleber
    """

    prefix = "tests/"
    file_name = "test_json.json"
    file_name = make_full_filename(prefix, file_name)
    a = {"1": np.array([1, 2, 3]), "2": np.int32(12)}
    save_json_to_file(a, file_name=file_name)
    loaded_a = load_json_from_file(file_name=file_name)
    assert list(loaded_a.keys()) == list(a.keys())


def test_search_subdir(): 
    """
    The test_search_subdir function tests the search_subdirs function.
    It does this by creating a directory with two subdirectories, each containing one file. 
    The test then calls the search_subdirs function on the parent directory and checks that it returns all three files and both directories.

    :return: A list of files and a list of directories.

    :doc-author: Trelent
    """


    search_dir = "tests/data/test_dir_search/"
    result_files, result_dirs = search_subdirs(dir_name = search_dir)
    
    assert len(result_files) == 4
    files = ['tests/data/test_dir_search/dir2/file1', 'tests/data/test_dir_search/dir2/dir1/file1','tests/data/test_dir_search/dir2/file2 ', 'tests/data/test_dir_search/dir1/file1']
    for name in files:
        assert name in result_files 
    
    reference_sub_dirs = ["tests/data/test_dir_search/dir1", "tests/data/test_dir_search/dir2", "tests/data/test_dir_search/dir2/dir1"]
    assert len(result_dirs) == 3
    for ref in reference_sub_dirs:
        assert ref in result_dirs 