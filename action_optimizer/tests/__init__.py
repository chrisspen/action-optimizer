"""Expose the test modules so `python -m unittest action_optimizer.tests` discovers them."""

import unittest

from . import test_autofill, test_optimizer

__all__ = ["test_autofill", "test_optimizer"]


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None):
    """Aggregate tests from the individual modules when the package is loaded."""
    suite = unittest.TestSuite()
    for module in (test_autofill, test_optimizer):
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
