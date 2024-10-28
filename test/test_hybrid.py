# import unittest
import pytest
from .utils_for_tests import ArgumentsTest, almostEqual

from pastasolver.pasta_solver import Pasta

@pytest.mark.parametrize("parameters", [
    ArgumentsTest("dummy_below_above", "../examples/hybrid/dummy_below_above.lp", "q0", 0.09678546088922922, 0.7882479221632422),
    ArgumentsTest("dummy_below_below", "../examples/hybrid/dummy_below_below.lp", "q0", 0.3032145391107708, 0.7180920158751786),
    ArgumentsTest("dummy_below_normalize", "../examples/hybrid/dummy_normalize.lp", "q0", 0.09307694498958728, 0.6330737873228345, normalize=True),
    ArgumentsTest("cold", "../examples/hybrid/cold.lp", "at_least_one_cold", 0.5888824281157551, 0.5888824281157551),
    ArgumentsTest("multi_interval", "../examples/hybrid/multi_interval.lp", "q0", 0.3246, 0.3246),
    ArgumentsTest("blood_pressure", "../examples/hybrid/blood_pressure.lp", "high_number_strokes", 0, 0.11729)
])
def test_hybrid(parameters : ArgumentsTest):
    pasta_solver = Pasta(parameters.filename, parameters.query, parameters.evidence, normalize_prob=parameters.normalize)
    lp, up = pasta_solver.inference()

    assert almostEqual(lp, parameters.expected_lp), f"{parameters.test_name}: wrong lower probability - E: {parameters.expected_lp}, F: {lp}"
    assert almostEqual(up, parameters.expected_up), f"{parameters.test_name}: wrong upper probability - E: {parameters.expected_up}, F: {up}"
