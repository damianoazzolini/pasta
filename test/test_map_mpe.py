import pytest

from .utils_for_tests import almost_equal, check_if_lists_equal

from pastasolver.pasta_solver import Pasta


@pytest.mark.parametrize("filename, query, test_name, expected_map_mpe, expected_atoms_list, upper", [
    ("../examples/map/bird_4_map.lp", "fly(1)", "map_bird_4", 0.0625, [
        ['bird(1)', 'bird(2)', 'not bird(3)', 'not bird(4)'],
        ['bird(1)', 'not bird(2)', 'bird(3)', 'not bird(4)'],
        ['bird(1)', 'not bird(2)', 'not bird(3)', 'bird(4)'],
        ['bird(1)', 'not bird(2)', 'not bird(3)', 'not bird(4)']
        ], False),
    ("../examples/map/gold_map.lp", "valuable(1)", "map_gold_lower", 0.09799999999999999, [['gold(3)', 'gold(1)']], False),
    ("../examples/map/gold_map.lp", "valuable(1)", "map_gold_upper", 0.13999999999999999, [['gold(3)', 'gold(1)']], True),
    ("../examples/map/smokes_map.lp", "smokes(c)", "map_smokes", 0.03125, [
        ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)'],
        ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'not e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)'],
        ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)'],
        ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)'],
        ['not e(a,b)', 'not e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)'],
        ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)']
        ],False),
    ("../examples/map/win_map.lp", "win", "map_win", 0.192, [['blue', 'red']], False),
    ("../examples/map/win_mpe.lp", "win", "mpe_win", 0.162, [['green', 'not red', 'blue', 'yellow']], False),
    ("../examples/map/win_mpe_disj.lp", "win", "mpe_win_disj_lower", 0.0324, [['not red', 'green', 'blue', 'yellow']], False),
    ("../examples/map/win_mpe_disj.lp", "win", "mpe_win_disj_upper", 0.1944, [['red', 'green', 'not blue', 'yellow']], True),
    ("../examples/map/simple_map_disj.lp", "win", "simple_map_disj_lower", 0.048, [['not a', 'b']], False),
    ("../examples/map/simple_map_disj.lp", "win", "simple_map_disj_upper", 0.32, [['a', 'b']], True),
    ("../examples/map/simple_map_disj_1.lp", "win", "simple_map_disj_lower", 0.056, [['a', 'b']], False),
    ("../examples/map/simple_map_disj_1.lp", "win", "simple_map_disj_upper", 0.56, [['a', 'b']], True)
])
def test_map_mpe(
    filename : str,
    query : str,
    test_name: str,
    expected_map_mpe: float,
    expected_atoms_list : 'list[list[str]]',
    upper : bool
    ):

    pasta_solver = Pasta(filename, query, consider_lower_prob=not upper)
    max_p, atoms_list = pasta_solver.map_inference()

    print(max_p, expected_map_mpe)
    if max_p > 0 and len(atoms_list) > 0:
        assert almost_equal(max_p, expected_map_mpe), test_name + f"{test_name}: wrong MAP/MPE - E: {expected_map_mpe}, F: {max_p}"
        assert check_if_lists_equal(atoms_list, expected_atoms_list), test_name + ": wrong atoms list"
