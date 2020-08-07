'''
docker compose 源码写日志
'''
import logging
from functools import wraps
import sys
import datetime
  
import _thread as thread
import sys
from collections import namedtuple
from itertools import cycle
from queue import Empty
from queue import Queue
from threading import Thread

from docker.errors import APIError

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("../log_config/%slog.txt"%(datetime.date.today()))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def decorator(function):
    @wraps(function)
    def inner(*args, **kwargs):
        try:
            print("当前运行方法", function.__name__)
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(f"{function.__name__} is error,here are details:{traceback.format_exc()}")
    return inner

class LogPrinter(object):
    """Print logs from many containers to a single output stream."""

    def __init__(self,
                 containers,
                 presenters,
                 event_stream,
                 output=sys.stdout,
                 cascade_stop=False,
                 log_args=None):
        self.containers = containers
        self.presenters = presenters
        self.event_stream = event_stream
        self.output = output
        self.cascade_stop = cascade_stop
        self.log_args = log_args or {}

    def run(self):
        if not self.containers:
            return

        queue = Queue()
        thread_args = queue, self.log_args
        thread_map = build_thread_map(self.containers, self.presenters, thread_args)
        start_producer_thread((
            thread_map,
            self.event_stream,
            self.presenters,
            thread_args))

        for line in consume_queue(queue, self.cascade_stop):
            remove_stopped_threads(thread_map)

            if self.cascade_stop:
                matching_container = [cont.name for cont in self.containers if cont.name == line]
                if line in matching_container:
                    # Returning the name of the container that started the
                    # the cascade_stop so we can return the correct exit code
                    return line

            if not line:
                if not thread_map:
                    # There are no running containers left to tail, so exit
                    return
                # We got an empty line because of a timeout, but there are still
                # active containers to tail, so continue
                continue

            self.write(line)
    @decorator
    def write(self, line):
        try:
            self.output.write(line)
        except UnicodeEncodeError:
            # This may happen if the user's locale settings don't support UTF-8
            # and UTF-8 characters are present in the log line. The following
            # will output a "degraded" log with unsupported characters
            # replaced by `?`
            self.output.write(line.encode('ascii', 'replace').decode())
        self.output.flush()