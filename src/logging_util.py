import logging
import time


class CustomFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_log_time = None
        self.base_fmt = '%(asctime)s%(elapsed_time)s %(name)s[%(levelname)s]: %(message)s'
        self.debug_fmt = '%(asctime)s%(elapsed_time)s %(name)s[%(levelname)s] %(funcName)s: %(message)s'
        self.datefmt = '%H:%M:%S'

    def format(self, record):
        current_time = time.time()
        if self.last_log_time is not None:
            elapsed_time = current_time - self.last_log_time
            record.elapsed_time = f" (+{elapsed_time:5.2f}s)"
        else:
            record.elapsed_time = ""
        self.last_log_time = current_time
        # Set format string based on log level
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_fmt
        else:
            self._style._fmt = self.base_fmt
        return super().format(record)


# Set up the logger
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
