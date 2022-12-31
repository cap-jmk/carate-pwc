import os

import subprocess


def test_cli_func():

    subprocess.run(["bash", "install.sh"])
    config_filepath = "tests/config/regression_test_config.py"
    subprocess.run(["carate", "-c", config_filepath])
    result_dir_content = os.listdir("tests/results/ZINC_test/data")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 2
    assert (
        r"ZINC_test_0.json" in result_dir_content
        and "ZINC_test_1.json" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ZINC_test/checkpoints")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".pt")]
    assert len(result_dir_content_data) == 4
    assert (
        "ZINC_test_CV-0_Epoch-1.pt" in result_dir_content_data
        and "ZINC_test_CV-0_Epoch-2.pt" in result_dir_content
        and "ZINC_test_CV-1_Epoch-1.pt" in result_dir_content
        and "ZINC_test_CV-1_Epoch-2.pt" in result_dir_content
    )
    result_dir_content = os.listdir("tests/results/ZINC_test/model_parameters")
    result_dir_content_data = [x for x in result_dir_content if x.endswith(".json")]
    assert len(result_dir_content_data) == 1
    assert "model_architecture.json" in result_dir_content_data
