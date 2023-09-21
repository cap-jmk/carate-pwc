from typing import Dict, Any
import datetime
import logging

from amarium.utils import prepare_file_name_saving


class MetricsLogger:
    """The class implements the Logger factory for basic metrics used in ML and
    deep Learning 
    
    :author: Julian M. Kleber
    """

    def __init__(self, save_dir: str) -> None:

        self.logger = logging.getLogger("metrics_logger")
        self.time_initialized = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.filename = prepare_file_name_saving(
            prefix=save_dir, file_name=f"{self.time_initialized}-metrics_logger", suffix=".log"
        )
        
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fileHandler = logging.FileHandler(self.filename)
        self.encoding = "utf-8"

        
        self.logger.setLevel(logging.DEBUG)

        self.fileHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.fileHandler)

        self.logger.info(f"Initialized Metrics Logger to {self.filename}")

    def log(self, metrics: Dict[str, Any], level: str = "info") -> None:
        """Logs the metrics using the self.logger."""
        for key, val in metrics.items():

            if level == "info":
                self.logger.info(self.basic_layout(key, val))
            elif level == "debug":
                self.logger.info(self.basic_layout(key, val))
            elif level == "warning":
                self.logger.warning(self.basic_layout(key, val))
            elif level == "error":
                self.logger.error(self.basic_layout(key, val))
            elif level == "critical":
                self.logger.critical(self.basic_layout(key, val))
            else:
                self.logger.warning(
                    f"Did not specify correct log level with {level}. Please either specify "
                    "debug, info, warning, error or critical"
                )
                self.logger.info(self.basic_layout(key, val))

    def basic_layout(self, metric: str, value: str) -> str:
        """
        Defines the basic logging layout for the MetricsLogger
        
        :param: metric: str: Name of the metric
        :paraM: value: str: Value of the metric 
        :author: Julian M. Kleber
        """
        return f"{metric} : {value}"
    
    def close_logger(self)->None: 
        """
        To limit logging to one file for multi run experiments
        the function closes the current logging file. 

        :author: Julian M. Kleber
        """
        handlers = self.logger.handlers
        for handler in handlers: 
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()
