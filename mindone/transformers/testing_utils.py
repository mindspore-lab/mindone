import unittest

from .utils import is_mindspore_available


def require_mindspore(test_case):
    """
    Decorator marking a test that requires MindSpore.

    These tests are skipped when MindSpore isn't installed.

    """
    return unittest.skipUnless(is_mindspore_available(), "test requires MindSpore")(test_case)
