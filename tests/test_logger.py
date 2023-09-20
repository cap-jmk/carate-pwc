import os 

from carate.logging.metrics_logger import MetricsLogger
from amarium.utils import delete_file, delete_all_files_in_dir, delete_empty_dir, make_full_filename

def test_metrics_logger():

    if os.path.isdir("tests/test_loggers/"):
        delete_all_files_in_dir("tests/test_loggers/")

    save_dir = "tests/test_loggers/"

    metricsLogger = MetricsLogger(save_dir)
    metricsLogger.log({"MAE":0.001}, "info")
    assert os.path.basename(metricsLogger.filename) in os.listdir("tests/test_loggers")

    delete_all_files_in_dir("tests/test_loggers/")
    delete_empty_dir("tests/test_loggers/")

    assert not os.path.isdir("test/test_loggers/")


