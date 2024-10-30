import argparse
import pytest

from pastasolver.pasta_solver import Pasta

from .utils_for_tests import ArgumentsTest, almost_equal


@pytest.mark.parametrize("parameters", [
    ArgumentsTest("alarm_calls_mary", "../examples/inference/alarm.lp", "calls(mary)", 0.196, 0.196),
    ArgumentsTest("bird_10_fly_1","../examples/inference/bird_10.lp", "fly(1)", 0.126953, 0.5),
    ArgumentsTest("bird_2_2_fly_1", "../examples/inference/bird_2_2.lp", "fly_1", 0.6, 0.7),
    ArgumentsTest("bird_4_fly_1", "../examples/inference/bird_4.lp", "fly(1)", 0.25, 0.5),
    ArgumentsTest("bird_4_different_fly_1", "../examples/inference/bird_4_different.lp", "fly(1)", 0.102222, 0.11),
    ArgumentsTest("disjunction", "../examples/inference/disjunction.lp", "f", 0.6, 0.8),
    ArgumentsTest("graph_coloring_qr", "../examples/inference/graph_coloring.lp", "qr", 0.03, 1),
    ArgumentsTest("path_path_1_4", "../examples/inference/path.lp", "path(1,4)", 0.266816, 0.266816),
    ArgumentsTest("shop_qr", "../examples/inference/shop.lp", "qr", 0.096, 0.3),
    ArgumentsTest("sick_sick", "../examples/inference/sick.lp", "sick", 0.2, 0.2384),
    ArgumentsTest("smoke_qry", "../examples/inference/smoke.lp", "qry", 0, 0.09),
    ArgumentsTest("smoke_3", "../examples/inference/smoke_3.lp", "qry", 0.3, 0.3),
    ArgumentsTest("viral_marketing_5_buy_5", "../examples/inference/viral_marketing_5.lp", "buy(5)", 0.2734, 0.29)
])
def test_approximate_inference(parameters : ArgumentsTest):
    pasta_solver = Pasta(parameters.filename, parameters.query, samples = 10_000)

    args = argparse.Namespace()
    args.rejection = parameters.rejection
    args.mh = parameters.mh
    args.gibbs = parameters.gibbs
    args.approximate_hybrid = False

    lp, up = pasta_solver.approximate_solve(args)

    assert almost_equal(lp, parameters.expected_lp), f"{parameters.test_name}: wrong lower probability - E: {parameters.expected_lp}, F: {lp}"
    assert almost_equal(up, parameters.expected_up), f"{parameters.test_name}: wrong upper probability - E: {parameters.expected_up}, F: {up}"
