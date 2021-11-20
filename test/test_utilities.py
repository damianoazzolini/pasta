# this import requires python 3.5 +
import importlib.util
from typing import Union

spec = importlib.util.spec_from_file_location("utilities", "../src/utilities.py")
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)

def test_is_number() -> Union[int,int]:
    failed = 0
    passed = 0

    print("test_is_number")

    n = 1.2
    expected = True
    res = utilities.is_number(n)
    if res != expected:
        print("Error: expected\n" + str(expected) + "\nfound\n" + str(res))
        failed = failed + 1
    else:
        passed = passed + 1

    n = 1
    expected = True
    res = utilities.is_number(n)
    if res != expected:
        print("Error: expected\n" + str(expected) + "\nfound\n" + str(res))
        failed = failed + 1
    else:
        passed = passed + 1

    n = 'a'
    expected = False
    res = utilities.is_number(n)
    if res != expected:
        print("Error: expected\n" + str(expected) + "\nfound\n" + str(res))
        failed = failed + 1
    else:
        passed = passed + 1

    if failed == 0:
        print("passed")
    else:
        print("failed")

    return failed,passed

def test_check_consistent_prob_fact() -> Union[int,int]:
    # TODO
    return 0,0

def test_extract_atom_between_brackets() -> Union[int,int]:
    # TODO
    return 0,0

def test_get_functor() -> Union[int,int]:
    # TODO
    return 0,0
    
def test_generate_dom_fact() -> Union[int,int]:
    failed = 0
    passed = 0

    print("test_generate_dom_fact")

    functor = "f"
    arguments = [['1', '3'], True]
    expected = "dom_f(1..3)."
    dom = utilities.generate_dom_fact(functor,arguments)

    if dom != expected:
        print("Error: expected\n" + expected + "\nfound\n" + str(dom))
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "g"
    arguments = [['a'], False]
    expected = "dom_g(a)."
    dom = utilities.generate_dom_fact(functor,arguments)

    if dom != expected:
        print("Error: expected\n" + expected + "\nfound\n" + str(dom))
        failed = failed + 1
    else:
        passed = passed + 1

    if failed == 0:
        print("passed")
    else:
        print("failed")

    return failed,passed
        
def test_utilities():
    n_function_tested = 0
    total = 0

    failed,passed = test_generate_dom_fact()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_get_functor()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_extract_atom_between_brackets()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_check_consistent_prob_fact()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_is_number()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    print("Tested " + str(n_function_tested) + " functions")
    print("Executed " + str(total) + " tests")
    print("Passed: " + str(passed))
    print("Failed: " + str(failed))

if __name__ == "__main__":
    test_utilities()