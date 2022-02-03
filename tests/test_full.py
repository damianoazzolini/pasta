from typing import Generator
import unittest

import importlib.util

import sys
from unittest import result
sys.path.append("../src/pasta/")

spec = importlib.util.spec_from_file_location("pasta", "../src/pasta/pasta.py")
past = importlib.util.module_from_spec(spec)
spec.loader.exec_module(past)


class TestBird(unittest.TestCase):
    def wrap_test(self, filename, query, evidence, test_name, expected_lp, expected_up, expected_abd = None):
        pasta_solver = past.Pasta(filename, query, evidence, 3, False, False)
        lp, up, abd = pasta_solver.solve()

        if expected_lp is not None and expected_up is not None:
            self.assertEqual(float(lp), expected_lp, test_name + ": wrong lower probability")
            self.assertEqual(float(up), expected_up, test_name + ": wrong upper probability")
        if expected_abd is not None:
            self.assertEqual(abd, expected_abd, test_name + ": wrong abductive explanation")


    def test_queries(self):
        self.wrap_test("../examples/bird_2_2.lp","fly_1", None, "bird_2_2_fly_1",0.6, 0.7)
        self.wrap_test("../examples/bird_4.lp","fly(1)", None, "bird_4_fly_1", 0.25, 0.5)
        self.wrap_test("../examples/bird_4_different.lp", "fly(1)", None, "bird_4_different_fly_1", 0.102222, 0.11)
        self.wrap_test("../examples/bird_4.lp","nofly(1)", None, "bird_4_nofly_1", 0.0, 0.25)
        self.wrap_test("../examples/bird_10.lp","fly(1)", None, "bird_10_fly_1", 0.126953, 0.5)
        self.wrap_test("../examples/bird_10.lp","nofly(1)", None, "bird_10_nofly_1", 0.0, 0.373046)
        self.wrap_test("../examples/path.lp","path(1,4)", None, "path_path_1_4", 0.266816, 0.266816)
        self.wrap_test("../examples/viral_marketing_5.lp", "buy(5)", None, "viral_marketing_5_buy_5", 0.2734, 0.29)
        self.wrap_test("../examples/bird_4_different.lp", "fly(1)","fly(2)", "bird_4_different_q_fly_1_e_fly_2", 0.073953, 0.113255)
        self.wrap_test("../examples/sick.lp", "sick", None, "sick_sick", 0.199, 0.2374)
        self.wrap_test("../examples/disjunction.lp", "f", None, "disjunction", 0.6, 0.8)

    def test_conditionals(self):
        self.wrap_test("../examples/conditionals/bird_4_cond.lp", "fly", None, "bird_4_cond_q_fly_1", 0.7, 1.0)
        self.wrap_test("../examples/conditionals/smokers.lp", "smk", None, "bird_4_cond_q_fly_1", 0.7, 0.70627)

    def test_deterministic_abduction(self):
        self.wrap_test("../examples/abduction/ex_1_det.lp", "q", None, "ex_1_det", None, None, [['q','abd_a','abd_b']])
        self.wrap_test("../examples/abduction/ex_2_det.lp", "q", None, "ex_2_det", None, None, [['q','abd_a','not_abd_b'],['q','not_abd_a','abd_b']])
        self.wrap_test("../examples/abduction/ex_3_det.lp", "q", None, "ex_3_det", None, None, [['q', 'abd_a', 'not_abd_b', 'not_abd_c', 'not_abd_d'], ['q', 'not_abd_a', 'abd_b', 'abd_c', 'not_abd_d']])

    def test_probabilistic_abduction(self):
        self.wrap_test("../examples/abduction/ex_1_prob.lp", "q", None, "ex_1_prob", 0.25, 0.25, [['abd_a','abd_b']])
        self.wrap_test("../examples/abduction/ex_2_prob.lp", "q", None, "ex_2_prob", 0.75, 0.75, [['abd_a', 'abd_b']])
        self.wrap_test("../examples/abduction/ex_3_prob.lp", "q", None, "ex_3_prob", 0.58, 0.58, [['abd_a', 'abd_b', 'abd_c', 'not_abd_d'], ['abd_a', 'abd_b', 'abd_c', 'abd_d']])
        self.wrap_test("../examples/abduction/ex_4_prob.lp", "q", None, "ex_4_prob", 0.648, 0.648, [['abd_c', 'abd_e']])


if __name__ == '__main__':
    unittest.main(buffer=True)
