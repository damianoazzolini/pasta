import unittest
import importlib.util
import sys

import utils_for_tests

sys.path.append("../pasta/")

spec = importlib.util.spec_from_file_location(
    "pasta", "../pasta/pasta_solver.py")
past = importlib.util.module_from_spec(spec)
spec.loader.exec_module(past)


class TestClassExactInference(unittest.TestCase):

    def wrap_test_hybrid(self, parameters : utils_for_tests.TestArguments):

        pasta_solver = past.Pasta(
            parameters.filename, parameters.query, parameters.evidence, normalize_prob=parameters.normalize)
        lp, up = pasta_solver.inference()

        self.assertTrue(
            utils_for_tests.almostEqual(lp, parameters.expected_lp),
            f"{parameters.test_name}: wrong lower probability - E: {parameters.expected_lp}, F: {lp}"
        )
        self.assertTrue(
            utils_for_tests.almostEqual(up, parameters.expected_up),
            f"{parameters.test_name}: wrong upper probability - E: {parameters.expected_up}, F: {up}"
        )

    def test_dummy_below_above(self):
        parameters = utils_for_tests.TestArguments(
            "dummy_below_above",
            "../examples/hybrid/dummy_below_above.lp",
            "q0",
            0.09678546088922922,
            0.7882479221632422
        )
        self.wrap_test_hybrid(parameters)

    def test_dummy_below_below(self):
        parameters = utils_for_tests.TestArguments(
            "dummy_below_below",
            "../examples/hybrid/dummy_below_below.lp",
            "q0",
            0.3032145391107708,
            0.7180920158751786
        )
        self.wrap_test_hybrid(parameters)

    def test_cold(self):
        parameters = utils_for_tests.TestArguments(
            "cold",
            "../examples/hybrid/cold.lp",
            "at_least_one_cold",
            0.5888824281157551,
            0.5888824281157551
        )
        self.wrap_test_hybrid(parameters)
        
    def test_multi_interval(self):
        parameters = utils_for_tests.TestArguments(
            "multi_interval",
            "../examples/hybrid/multi_interval.lp",
            "q0",
            0.3246,
            0.3246
        )
        self.wrap_test_hybrid(parameters)

    def test_blood_pressure(self):
        parameters = utils_for_tests.TestArguments(
            "blood_pressure",
            "../examples/hybrid/blood_pressure.lp",
            "high_number_strokes",
            0,
            0.11729
        )
        self.wrap_test_hybrid(parameters)

if __name__ == '__main__':
    unittest.main(buffer=True)
