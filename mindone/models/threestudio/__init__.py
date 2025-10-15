import logging

__modules__ = {}


def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


logger = logging.getLogger("")
info = logger.info
debug = logger.debug

from . import data, models, systems
