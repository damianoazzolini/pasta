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
    def wrap_test(self, filename, query, evidence, test_name, expected_lp, expected_up):
        pasta_solver = past.Pasta(filename, query, evidence, 3, False, False)
        lp, up, _ = pasta_solver.solve()

        self.assertEqual(float(lp), expected_lp, test_name + ": wrong lower probability")
        self.assertEqual(float(up), expected_up, test_name + ": wrong upper probability")

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

if __name__ == '__main__':
    unittest.main(buffer=True)
