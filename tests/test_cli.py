import os
import shutil

import subprocess

from amarium.utils import search_subdirs

from tests.utils import check_result_files


def test_cli_func():
    if os.path.isdir("tests/data"):
        shutil.rmtree("tests/data")

    subprocess.run(["bash", "install.sh"])
    config_filepath = "tests/config/regression_test_config_override.py"
    result = subprocess.run(
        ["carate", "-c", config_filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errs = result.stderr.decode()
    run_title = "ZINC_test"
    data_set_name = "ZINC_test"
    result_dir = f"tests/results/{run_title}"
    check_result_files(
        result_dir=result_dir, run_title=run_title, data_set_name=data_set_name
    )
