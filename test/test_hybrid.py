import unittest

import pytest

import importlib.util

import utils_for_tests

import sys
sys.path.append("../pasta/")

spec = importlib.util.spec_from_file_location(
    "pasta", "../pasta/pasta_solver.py")
past = importlib.util.module_from_spec(spec)
spec.loader.exec_module(past)


class TestClassExactInference(unittest.TestCase):

    def wrap_test_hybrid(self,
                        filename: str,
                        query: str,
                        evidence: str,
                        test_name: str,
                        expected_lp: float,
                        expected_up: float,
                        normalize: bool = False):

        if evidence is None:
            evidence = ""

        pasta_solver = past.Pasta(
            filename, query, evidence, normalize_prob=normalize)
        lp, up = pasta_solver.inference()

        self.assertTrue(utils_for_tests.almostEqual(
            lp, expected_lp, 5), test_name + ": wrong lower probability")
        self.assertTrue(utils_for_tests.almostEqual(
            up, expected_up, 5), test_name + ": wrong upper probability")

    def test_dummy_below_above(self):
        self.wrap_test_hybrid(
            "../examples/hybrid/dummy_below_above.lp", 
            "q0", 
            "", 
            "dummy_below_above", 
            0.09678546088922922,
            0.7882479221632422
        )

    def test_dummy_below_below(self):
        self.wrap_test_hybrid(
            "../examples/hybrid/dummy_below_below.lp", 
            "q0", 
            "", 
            "dummy_below_below",
            0.3032145391107708,
            0.7180920158751786
        )

    def test_cold(self):
        self.wrap_test_hybrid(
            "../examples/hybrid/cold.lp", 
            "at_least_one_cold",
            "", 
            "dummy_below_below",
            0.5888824281157551,
            0.5888824281157551
        )

   
if __name__ == '__main__':
    unittest.main(buffer=True)
