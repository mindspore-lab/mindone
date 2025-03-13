import unittest

from .utils import is_mindspore_available


def require_mindspore(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    return unittest.skipUnless(is_mindspore_available(), "test requires PyTorch")(test_case)
