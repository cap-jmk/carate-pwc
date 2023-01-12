"""The runner object needs to be absolutely 
bullet proof in initialization. One false initialization
leads to weird results 

:author: Julian M. Kleber
"""
from carate.run import RunInitializer


def test_runner():
    # Regression
    config_filepath = "tests/config/regression_test_config.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    assert runner.dataset_name == "ZINC_test"
    assert (
        str(runner.Evaluation)
        == "Evaluation for cgc_regression with the Default evaluation"
    )
