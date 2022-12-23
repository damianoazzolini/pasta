import sys
import argparse
import importlib.util
import pytest
import unittest

import utils_for_tests



sys.path.append("../pasta/")

spec = importlib.util.spec_from_file_location(
    "pasta", "../pasta/pasta_solver.py")
past = importlib.util.module_from_spec(spec)
spec.loader.exec_module(past)


class TestClassApproximateInference(unittest.TestCase):

    def wrap_test_approximate_inference(self,
                                        filename: str,
                                        query: str,
                                        evidence: str,
                                        test_name: str,
                                        expected_lp: float,
                                        expected_up: float,
                                        samples: int = 5000):

        if evidence is None:
            evidence = ""

        pasta_solver = past.Pasta(filename, query, samples=samples)
        # ps = pasta_solver.Pasta(filename, query, samples)

        args = argparse.Namespace()
        args.rejection = False
        args.mh = False
        args.gibbs = False

        lp, up = pasta_solver.approximate_solve(args)

        self.assertTrue(utils_for_tests.almostEqual(lp, expected_lp, 0.02),
                        test_name + ": wrong lower probability")
        self.assertTrue(utils_for_tests.almostEqual(up, expected_up, 0.02),
                        test_name + ": wrong upper probability")

    def test_alarm_calls_mary(self):
        self.wrap_test_approximate_inference("../examples/inference/alarm.lp","calls(mary)", "", "alarm_calls_mary",0.7, 0.7)


    def test_bird_10_fly_1(self):
        self.wrap_test_approximate_inference("../examples/inference/bird_10.lp","fly(1)", "", "bird_10_fly_1", 0.126953, 0.5)


    def test_bird_2_2_fly_1(self):
        self.wrap_test_approximate_inference("../examples/inference/bird_2_2.lp", "fly_1", "", "bird_2_2_fly_1", 0.6, 0.7)


    def test_bird_4_fly_1(self):
        self.wrap_test_approximate_inference("../examples/inference/bird_4.lp","fly(1)", "", "bird_4_fly_1", 0.25, 0.5)


    def test_bird_4_different_fly_1(self):
        self.wrap_test_approximate_inference("../examples/inference/bird_4_different.lp", "fly(1)", "", "bird_4_different_fly_1", 0.102222, 0.11)


    def test_clique_in_1_exit(self):
        with pytest.raises(SystemExit):
            self.wrap_test_approximate_inference("../examples/inference/clique.lp", "in(1)", "", "clique_in_1", 0.4666666666666667, 0.9333333333333333)


    def test_disjunction(self):
        self.wrap_test_approximate_inference("../examples/inference/disjunction.lp", "f", "", "disjunction", 0.6, 0.8)


    def test_evidence_certain(self):
        self.wrap_test_approximate_inference("../examples/inference/evidence_certain.lp", "qr", "", "evidence_certain", 0, 1)


    def test_graph_coloring_qr(self):
        self.wrap_test_approximate_inference("../examples/inference/graph_coloring.lp", "qr", "", "graph_coloring_qr", 0.03, 1.0)


    def test_path_path_1_4(self):
       self.wrap_test_approximate_inference("../examples/inference/path.lp","path(1,4)", "", "path_path_1_4", 0.266816, 0.266816)


    def test_shop_qr(self):
        self.wrap_test_approximate_inference("../examples/inference/shop.lp", "qr", "", "shop_qr", 0.0625, 0.5)


    def test_sick_sick(self):
        self.wrap_test_approximate_inference("../examples/inference/sick.lp", "sick", "", "sick_sick", 0.2, 0.2384)


    def test_smoke_qry(self):
        self.wrap_test_approximate_inference("../examples/inference/smoke.lp", "qry", "", "smoke_qry", 0, 0.09)



    def test_smoke_2_qr_exit(self):
        with pytest.raises(SystemExit):
            self.wrap_test_approximate_inference("../examples/inference/smoke_2.lp", "qr", "", "smoke_2_qr", 0.055408970976253295, 0.13398746701846967)

    
    def test_smoke_3(self):
        self.wrap_test_approximate_inference("../examples/inference/smoke_3.lp", "qry", "", "smoke_3", 0.3, 0.3)


    def test_viral_marketing_5_buy_5(self):
        self.wrap_test_approximate_inference("../examples/inference/viral_marketing_5.lp", "buy(5)", "", "viral_marketing_5_buy_5", 0.2734, 0.29)


if __name__ == '__main__':
    unittest.main(buffer=True)
