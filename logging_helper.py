"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import os
import datetime
import logging
from colorlog import ColoredFormatter
import colored_traceback.auto 

FILE_FORMAT = "%(asctime)s %(name)-30s %(levelname)-8s %(message)s"
FILE_DATE_FORMAT = "%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(log_color)s%(message)s%(reset)s"

loglv = 10

logging.addLevelName(21, 'EVAL')

def _eval(self, msg, *args, **kwargs):
    self.log(25, msg, *args, **kwargs)

logging.Logger.eval = _eval


console_handle = None

root_logger = logging.getLogger("")
package_logger = logging.getLogger(__name__.split(".")[0])
logger = logging.getLogger(__name__)



def _log_exception(exc_type, exc_value, exc_traceback):
    if console_handle:
        root_logger.removeHandler(console_handle)
    if not issubclass(exc_type, KeyboardInterrupt):
        root_logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
    if console_handle:
        root_logger.addHandler(console_handle)

def setup(log_dir = None, filename = None):
    if log_dir is not None:
        if log_dir != "" and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if filename is None:
            date = datetime.datetime.now()
            filename = "log_{date:%Y_%m_%d-%H_%M_%S}.txt".format(date=date)
        file_path = os.path.join(log_dir, filename)


        file = logging.FileHandler(file_path, mode="a")
        formatter = logging.Formatter()

        file.setLevel(10)
        file.setFormatter(formatter)
        root_logger.addHandler(file)
        
        global console_handle  
        if console_handle is None:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = ColoredFormatter(
                    "%(log_color)s%(message)s%(reset)s",
                    datefmt=None,
                    reset=True,
                    log_colors={
                        'DEBUG':    'green',
                        'MINFO':    'green',
                        'INFO':     'green',
                        'EVAL':     'green',
                        'WARNING':  'yellow',
                        'ERROR':    'red,bold',
                        'CRITICAL': 'red,bg_white',
                    },
                    secondary_log_colors={},
                    style='%'
                )
            console.setFormatter(formatter)
            root_logger.addHandler(console)
            console_handle = console
        sys.excepthook = _log_exception
        package_logger.setLevel(logging.INFO)