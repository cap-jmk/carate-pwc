import os 

from carate.logging.metrics_logger import MetricsLogger
from amarium.utils import delete_file

def test_metrics_logger():

    delete_all_files()

    save_dir = "tests/test_loggers/"

    metricsLogger = MetricsLogger(save_dir)
    metricsLogger.log({"MAE":0.001}, "info")
    assert metricsLogger.filename in os.listdir("tests/test_loggers")

    delete_file(metricsLogger.filename)

    assert metricsLogger.filename not in os.listdir("tests/test_loggers")


