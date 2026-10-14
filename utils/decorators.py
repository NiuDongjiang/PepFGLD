
import functools
from concurrent import futures
TimeoutError = futures.TimeoutError


def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

class timeout:
    __executor = futures.ThreadPoolExecutor(1)

    def __init__(self, seconds):
        self.seconds = seconds

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = timeout.__executor.submit(func, *args, **kw)
            return future.result(timeout=self.seconds)
        return wrapper