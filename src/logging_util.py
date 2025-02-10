import logging
import time


class CustomFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_log_time = None
        self._style._fmt = '%(asctime)s\033[33m%(elapsed_time)s\033[0m %(name)s[\033[94m%(levelname)s\033[0m]:%(message)s'
        self.datefmt = '%H:%M:%S'

    def format(self, record):
        current_time = time.time()
        if self.last_log_time is not None:
            elapsed_time = current_time - self.last_log_time
            record.elapsed_time = f" (+{elapsed_time:5.2f}s)"
        else:
            record.elapsed_time = ""
        self.last_log_time = current_time
        return super().format(record)


# Set up the logger
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
