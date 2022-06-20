import unittest

import importlib.util

import sys
sys.path.append("../src/pasta/")

spec = importlib.util.spec_from_file_location("pasta", "../src/pasta/pasta_solver.py")
past = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(past)  # type: ignore


# TODO: refactor, one test class for inference type and a method
# for every program tested
class TestClass(unittest.TestCase):
    @staticmethod
    def almostEqual(a : float, b : float, digits : int) -> bool:
        epsilon = 10 ** -digits
        if b == 0:
            return a == b
        else:
            return abs(a/b - 1) < epsilon

    def setup_solver(self, filename : str, query : str, evidence : str = ""):
        return past.Pasta(filename, query, evidence)

    def wrap_test_exact_inference(self, 
        filename : str, 
        query : str, 
        evidence : str, 
        test_name : str, 
        expected_lp : float, 
        expected_up : float, 
        expected_abd : 'list[str]' = []):
        if evidence is None:
            evidence = ""
        pasta_solver = past.Pasta(filename, query, evidence, 3, False, False)
        lp, up = pasta_solver.inference()

        self.assertTrue(self.almostEqual(lp,expected_lp,5), test_name + ": wrong lower probability")
        self.assertTrue(self.almostEqual(up,expected_up,5), test_name + ": wrong upper probability")
            # self.assertEqual(float(lp), expected_lp, test_name + ": wrong lower probability")
            # self.assertEqual(float(up), expected_up, test_name + ": wrong upper probability")
        # if expected_abd is not None:
        #     self.assertCountEqual(abd, expected_abd, test_name + ": wrong abductive explanation")

    def wrap_test_abduction(self, 
        filename : str, 
        query : str, 
        evidence  : str, 
        test_name  : str, 
        expected_lp : float, 
        expected_up : float, 
        expected_abd : 'list [str]'):
        pasta_solver = past.Pasta(filename, query, evidence, 3, False, False)
        lp, up, _ = pasta_solver.abduction()

        if expected_lp is not None and expected_up is not None:
            self.assertTrue(self.almostEqual(lp, expected_lp, 5),
                            test_name + ": wrong lower probability")
            self.assertTrue(self.almostEqual(up, expected_up, 5),
                            test_name + ": wrong upper probability")
            # self.assertEqual(float(lp), expected_lp, test_name + ": wrong lower probability")
            # self.assertEqual(float(up), expected_up, test_name + ": wrong upper probability")
        # if expected_abd is not None:
        #     self.assertCountEqual(abd, expected_abd, test_name + ": wrong abductive explanation")



    def test_queries(self):
        self.wrap_test_exact_inference("../examples/inference/bird_2_2.lp","fly_1", "", "bird_2_2_fly_1",0.6, 0.7)
        self.wrap_test_exact_inference("../examples/inference/bird_4.lp","fly(1)", "", "bird_4_fly_1", 0.25, 0.5)
        self.wrap_test_exact_inference("../examples/inference/bird_4_different.lp", "fly(1)", "", "bird_4_different_fly_1", 0.102222, 0.11)
        self.wrap_test_exact_inference("../examples/inference/bird_4.lp","nofly(1)", "", "bird_4_nofly_1", 0.0, 0.25)
        self.wrap_test_exact_inference("../examples/inference/bird_10.lp","fly(1)", "", "bird_10_fly_1", 0.126953, 0.5)
        self.wrap_test_exact_inference("../examples/inference/bird_10.lp","nofly(1)", "", "bird_10_nofly_1", 0.0, 0.373046)
        self.wrap_test_exact_inference("../examples/inference/path.lp","path(1,4)", "", "path_path_1_4", 0.266816, 0.266816)
        self.wrap_test_exact_inference("../examples/inference/viral_marketing_5.lp", "buy(5)", "", "viral_marketing_5_buy_5", 0.2734, 0.29)
        self.wrap_test_exact_inference("../examples/inference/bird_4_different.lp", "fly(1)","fly(2)", "bird_4_different_q_fly_1_e_fly_2", 0.073953, 0.113255)
        # self.wrap_test_exact_inference("../examples/inference/sick.lp", "sick", None, "sick_sick", 0.199, 0.2384)
        self.wrap_test_exact_inference("../examples/inference/disjunction.lp", "f", "", "disjunction", 0.6, 0.8)
        self.wrap_test_exact_inference("../examples/inference/certain_fact.lp", "a1", "", "certain_fact", 1, 1)
        self.wrap_test_exact_inference("../examples/inference/evidence_certain.lp", "qr", "ev", "evidence_certain", 1, 1)


    def test_conditionals(self):
        self.wrap_test_exact_inference("../examples/conditionals/bird_4_cond.lp", "fly", "", "bird_4_cond_q_fly_1", 0.7, 1.0)
        self.wrap_test_exact_inference(
            "../examples/conditionals/smokers.lp", "smk", "", "bird_4_cond_q_fly_1", 0.7, 0.70627)
    

    # MAP/MPE. TODO: test with lower and upper options 
    def test_mpe_win(self):
        solver = self.setup_solver("../examples/map/win_mpe.lp", "win")
        max_p, atoms_list = solver.map_inference()
        self.assertAlmostEqual(max_p, 0.162)
        self.assertListEqual(atoms_list, ['not red', 'green', 'blue', 'yellow'])
    
    def test_map_win(self):
        solver = self.setup_solver("../examples/map/win_map.lp", "win")
        max_p, atoms_list = solver.map_inference()
        self.assertAlmostEqual(max_p, 0.192)
        self.assertListEqual(atoms_list, ['red','blue'])

    # def test_deterministic_abduction(self):
    #     self.wrap_test_abduction("../examples/abduction/ex_1_det.lp", "query", None, "ex_1_det", 1, 1, [['abd_a', 'abd_b', 'q']])
    #     self.wrap_test_abduction("../examples/abduction/ex_2_det.lp", "query", None, "ex_2_det", 1, 1, [['q','abd_a','not_abd_b'],['q','not_abd_a','abd_b']])
    #     self.wrap_test_abduction("../examples/abduction/ex_3_det.lp", "query", None, "ex_3_det", 1, 1, [['q', 'abd_a', 'not_abd_b', 'not_abd_c', 'not_abd_d'], ['q', 'not_abd_a', 'abd_b', 'abd_c', 'not_abd_d']])
    #     self.wrap_test_abduction("../examples/abduction/ex_4_det.lp", "query", None, "ex_4_det", 1, 1, [['abd_a(1)', 'not_abd_b', 'not_abd_c','not_abd_d','q'], ['not_abd_a(1)','abd_b','abd_c','not_abd_d','q']])
    #     self.wrap_test_abduction("../examples/abduction/smokes_det.lp", "smokes(c)", None, "smokes_det", 1, 1, [['abd_e(b,c)', 'not_abd_e(a,b)', 'not_abd_e(e,c)', 'not_abd_e(d,e)', 'not_abd_e(a,d)', 'q'], ['abd_e(e,c)', 'abd_e(d,e)', 'not_abd_e(a,b)', 'not_abd_e(a,d)', 'not_abd_e(b,c)', 'q']])
    #     self.wrap_test_abduction("../examples/abduction/ex_5_det.lp", "qr", None, "smokes_det", 1, 1, [['abd_a', 'abd_b', 'abd_c'], ['abd_a', 'abd_b', 'abd_d','abd_e']])

    # def test_probabilistic_abduction(self):
    #     self.wrap_test_abduction("../examples/abduction/ex_1_prob.lp", "query", None, "ex_1_prob", 0.25, 0.25, [['abd_a','abd_b']])
    #     self.wrap_test_abduction("../examples/abduction/ex_2_prob.lp", "query", None, "ex_2_prob", 0.75, 0.75, [['abd_a', 'abd_b']])
    #     self.wrap_test_abduction("../examples/abduction/ex_3_prob.lp", "query", None, "ex_3_prob", 0.58, 0.58, [
    #                                    ['abd_a', 'abd_b', 'abd_c', 'not_abd_d'], ['abd_a', 'abd_b', 'abd_c', 'abd_d']])
    #     self.wrap_test_abduction("../examples/abduction/ex_4_prob.lp", "query", None, "ex_4_prob", 0.648, 0.648, [['abd_c', 'abd_e']])


if __name__ == '__main__':
    unittest.main(buffer=True)
