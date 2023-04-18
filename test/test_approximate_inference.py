import argparse
import unittest

import utils_for_tests

from pasta.pasta_solver import Pasta

class TestClassApproximateInference(unittest.TestCase):

    def wrap_test_approximate_inference(self, parameters : utils_for_tests.TestArguments):
        pasta_solver = Pasta(parameters.filename, parameters.query, samples = 5000)

        args = argparse.Namespace()
        args.rejection = parameters.rejection
        args.mh = parameters.mh
        args.gibbs = parameters.gibbs

        lp, up = pasta_solver.approximate_solve(args)

        self.assertTrue(
            utils_for_tests.almostEqual(lp, parameters.expected_lp),
            f"{parameters.test_name}: wrong lower probability - E: {parameters.expected_lp}, F: {lp}"
        )
        self.assertTrue(
            utils_for_tests.almostEqual(up, parameters.expected_up),
            f"{parameters.test_name}: wrong upper probability - E: {parameters.expected_up}, F: {up}"
        )


    def test_alarm_calls_mary(self):
        parameters = utils_for_tests.TestArguments(
            "alarm_calls_mary",
            "../examples/inference/alarm.lp",
            "calls(mary)",
            0.196,
            0.196
        )
        self.wrap_test_approximate_inference(parameters)

    def test_bird_10_fly_1(self):
        parameters = utils_for_tests.TestArguments(
            "bird_10_fly_1",
            "../examples/inference/bird_10.lp",
            "fly(1)",
            0.126953,
            0.5
        )
        self.wrap_test_approximate_inference(parameters)

    def test_bird_2_2_fly_1(self):
        parameters = utils_for_tests.TestArguments(
            "bird_2_2_fly_1",
            "../examples/inference/bird_2_2.lp",
            "fly_1",
            0.6,
            0.7
        )
        self.wrap_test_approximate_inference(parameters)

    def test_bird_4_fly_1(self):
        parameters = utils_for_tests.TestArguments(
            "bird_4_fly_1",
            "../examples/inference/bird_4.lp",
            "fly(1)",
            0.25,
            0.5
        )
        self.wrap_test_approximate_inference(parameters)

    def test_bird_4_different_fly_1(self):
        parameters = utils_for_tests.TestArguments(
            "bird_4_different_fly_1",
            "../examples/inference/bird_4_different.lp",
            "fly(1)",
            0.102222,
            0.11
        )
        self.wrap_test_approximate_inference(parameters)

    def test_disjunction(self):
        parameters = utils_for_tests.TestArguments(
            "disjunction",
            "../examples/inference/disjunction.lp",
            "f",
            0.6,
            0.8
        )
        self.wrap_test_approximate_inference(parameters)

    # def test_evidence_certain(self):
    #     parameters = utils_for_tests.TestArguments(
    #         "evidence_certain",
    #         "../examples/inference/evidence_certain.lp",
    #         "qr",
    #         0,
    #         1
    #     )
    #     self.wrap_test_approximate_inference(parameters)

    def test_graph_coloring_qr(self):
        parameters = utils_for_tests.TestArguments(
            "graph_coloring_qr",
            "../examples/inference/graph_coloring.lp",
            "qr",
            0.03,
            1
        )
        self.wrap_test_approximate_inference(parameters)

    def test_path_path_1_4(self):
        parameters = utils_for_tests.TestArguments(
            "path_path_1_4",
            "../examples/inference/path.lp",
            "path(1,4)",
            0.266816,
            0.266816
        )
        self.wrap_test_approximate_inference(parameters)

    def test_shop_qr(self):
        parameters = utils_for_tests.TestArguments(
            "shop_qr",
            "../examples/inference/shop.lp",
            "qr",
            0.0625,
            0.5
        )
        self.wrap_test_approximate_inference(parameters)

    def test_sick_sick(self):
        parameters = utils_for_tests.TestArguments(
            "sick_sick",
            "../examples/inference/sick.lp",
            "sick",
            0.2,
            0.2384
        )
        self.wrap_test_approximate_inference(parameters)

    def test_smoke_qry(self):
        parameters = utils_for_tests.TestArguments(
            "smoke_qry",
            "../examples/inference/smoke.lp",
            "qry",
            0,
            0.09
        )
        self.wrap_test_approximate_inference(parameters)

    def test_smoke_3(self):
        parameters = utils_for_tests.TestArguments(
            "smoke_3",
            "../examples/inference/smoke_3.lp",
            "qry",
            0.3,
            0.3
        )
        self.wrap_test_approximate_inference(parameters)

    def test_viral_marketing_5_buy_5(self):
        parameters = utils_for_tests.TestArguments(
            "viral_marketing_5_buy_5",
            "../examples/inference/viral_marketing_5.lp",
            "buy(5)",
            0.2734,
            0.29
        )
        self.wrap_test_approximate_inference(parameters)

    # # test MH sampling
    # def test_mh_bird_4_fly_1(self):
    #     self.wrap_test_approximate_inference("../examples/inference/bird_4.lp","fly(1)", "", "bird_4_fly_1", 0.25, 0.5)

    
    # # test Gibbs sampling
    # def test_gibbs_bird(self):
    #     pass
    
    # # test rejection sampling
    # def test_rej_bird(self):
    #     pass

    
if __name__ == '__main__':
    unittest.main(buffer=True)
