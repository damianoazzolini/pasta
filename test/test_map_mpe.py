import unittest

import pytest

import importlib.util

import utils_for_tests

from pastasolver.pasta_solver import Pasta

# t_utils = __import__('test_utils')

            


class TestClassMapMpe(unittest.TestCase):

    def wrap_test_map_mpe(self,
        filename : str,
        query : str,
        evidence : str,
        test_name: str,
        expected_map_mpe: float,
        expected_atoms_list : 'list[list[str]]',
        upper : bool = False):

        pasta_solver = Pasta(filename, query, evidence)
        max_p, atoms_list = pasta_solver.map_inference()

        if upper is True:
            max_p, atoms_list = pasta_solver.upper_mpe_inference()


        if max_p is not None and atoms_list is not None:
            self.assertTrue(utils_for_tests.almostEqual(max_p, expected_map_mpe, 5), test_name + ": wrong MAP/MPE")
    
            self.assertTrue(utils_for_tests.check_if_lists_equal(atoms_list, expected_atoms_list), test_name + ": wrong atoms list")


    def test_map_bird_4(self):
        self.wrap_test_map_mpe("../examples/map/bird_4_map.lp", "fly(1)", "", "map_bird_4", 0.0625, [['bird(1)', 'bird(2)', 'not bird(3)', 'not bird(4)'], 
        ['bird(1)', 'not bird(2)', 'bird(3)', 'not bird(4)'], ['bird(1)', 'not bird(2)', 'not bird(3)', 'bird(4)'], 
        ['bird(1)', 'not bird(2)', 'not bird(3)', 'not bird(4)']])


    def test_map_gold(self):
        self.wrap_test_map_mpe("../examples/map/gold_map.lp", "valuable(1)", "", "map_gold", 0.060000000000000005, [['not gold(3)', 'gold(1)']])
        

    def test_map_gold_upper(self):
        with pytest.raises(SystemExit):
            self.wrap_test_map_mpe("../examples/map/gold_map.lp", "valuable(1)", "", "map_gold", 0.13999999999999999, [['gold(1)', 'gold(3)']], True)
        
    
    def test_map_smokes(self):
        self.wrap_test_map_mpe("../examples/map/smokes_map.lp", "smokes(c)", "", "map_smokes", 0.03125, [['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)'], 
        ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)'], ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'not e(e,c)'], 
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)'], ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)'], 
        ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)'], ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)'], 
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)'], ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)'], 
        ['not e(a,b)', 'not e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)'], ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)']])


    def test_map_win(self):
        self.wrap_test_map_mpe("../examples/map/win_map.lp", "win", "", "map_win", 0.192, [['blue', 'red']])
       

    def test_mpe_win(self):
        self.wrap_test_map_mpe("../examples/map/win_mpe.lp", "win", "", "mpe_win", 0.162, [['green', 'not red', 'blue', 'yellow']])


      


if __name__ == '__main__':
    unittest.main(buffer=True)