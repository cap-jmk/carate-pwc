import os
import shutil

import subprocess

from amarium.utils import search_subdirs


def test_cli_func():

    if os.path.isdir("tests/data"):
        shutil.rmtree("tests/data")
    if os.path.isdir("tests/results"):
        shutil.rmtree("tests/results")
    subprocess.run(["bash", "install.sh"])
    config_filepath = "tests/config/regression_test_config_override.py"
    result = subprocess.run(
        ["carate", "-c", config_filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errs = result.stderr.decode()
    print(errs)
    # check result files
    result_files, result_dirs = search_subdirs(dir_name="tests/results/ZINC_test/data")
    reference_dirs = [
        "tests/results/ZINC_test/data/CV_0",
        "tests/results/ZINC_test/data/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ZINC_test/data/CV_0/ZINC_test.json",
        "tests/results/ZINC_test/data/CV_1/ZINC_test.json",
    ]
    for name in result_files:
        assert name in reference_files

    # check result checkpoints
    result_files, result_dirs = search_subdirs(
        dir_name="tests/results/ZINC_test/checkpoints"
    )

    reference_dirs = [
        "tests/results/ZINC_test/checkpoints/CV_0",
        "tests/results/ZINC_test/checkpoints/CV_1",
    ]
    assert len(result_dirs) == 2
    for dir_name in result_dirs:
        assert dir_name in reference_dirs

    assert len(result_files) == 2
    reference_files = [
        "tests/results/ZINC_test/checkpoints/CV_0/ZINC_test.tar",
        "tests/results/ZINC_test/checkpoints/CV_1/ZINC_test.tar",
    ]
    for name in result_files:
        assert name in reference_files

    result_dir_content = os.listdir("tests/results/ZINC_test/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data
