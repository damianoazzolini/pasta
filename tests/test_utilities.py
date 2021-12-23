# this import requires python 3.5 +
import importlib.util
from typing import Union

spec = importlib.util.spec_from_file_location("utilities", "../src/pasta/utilities.py")
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

def test_generate_model_clause() -> Union[int,int]:
    # TODO
    return 0,0

def test_generate_dom_fact() -> Union[int,int]:
    failed = 0
    passed = 0

    print("test_generate_dom_fact")

    functor = "f"
    arguments = [['1', '3'], True]
    expected_dom = "dom_f(1..3)."
    expected_args = "I"
    dom,args = utilities.generate_dom_fact(functor,arguments)

    if dom != expected_dom:
        print("Error: expected\n" + expected_dom + "\nfound\n" + str(dom))
    if args != expected_args:
        print("Error: expected\n" + expected_args + "\nfound\n" + str(args))
        
    if dom != expected_dom or args != expected_args:
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "g"
    arguments = [['a'], False]
    expected_dom = "dom_g(a)."
    expected_args = "a"
    dom,args = utilities.generate_dom_fact(functor,arguments)

    if dom != expected_dom:
        print("Error: expected\n" + expected_dom + "\nfound\n" + str(dom))
    if args != expected_args:
        print("Error: expected\n" + expected_args + "\nfound\n" + str(args))
        
    if dom != expected_dom or args != expected_args:
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "g"
    arguments = [['a','b'], False]
    expected_dom = "dom_g(a,b)."
    expected_args = "a,b"
    dom,args = utilities.generate_dom_fact(functor,arguments)

    if dom != expected_dom:
        print("Error: expected\n" + expected_dom + "\nfound\n" + str(dom))
    if args != expected_args:
        print("Error: expected\n" + expected_args + "\nfound\n" + str(args))
        
    if dom != expected_dom or args != expected_args:
        failed = failed + 1
    else:
        passed = passed + 1

    if failed == 0:
        print("passed")
    else:
        print("failed")

    return failed,passed

def test_generate_generator() -> Union[int,int]:
    failed = 0
    passed = 0

    print("test_generate_generator")

    functor = "f"
    arguments = [['1', '3'], True]
    args = "I"
    expected_gen = "0{v_f_(I):dom_f(I)}3."
    expected_list = ['f(1,693):-v_f_(1).', 'not_f(1,693):- not v_f_(1).', 'f(2,693):-v_f_(2).', 'not_f(2,693):- not v_f_(2).', 'f(3,693):-v_f_(3).', 'not_f(3,693):- not v_f_(3).']
    res_gen, res_list = utilities.generate_generator(functor,args,arguments,0.5,1000)

    if res_gen != expected_gen:
        print("Error: expected\n" + str(expected_gen) + "\nfound\n" + str(res_gen))
    if res_list != expected_list:
        print("Error: expected\n" + str(expected_list) + "\nfound\n" + str(res_list))       
    
    if res_gen != expected_gen or res_list != expected_list:
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "g"
    arguments = [['a'], False]
    args = "a"
    expected_gen = "0{v_g_(a):dom_g(a)}1."
    expected_list = ['g(a,1609):-v_g_(a).', 'not_g(a,1609):- not v_g_(a).']
    res_gen, res_list = utilities.generate_generator(functor,args,arguments,0.2,1000)

    if res_gen != expected_gen:
        print("Error: expected\n" + str(expected_gen) + "\nfound\n" + str(res_gen))
    if res_list != expected_list:
        print("Error: expected\n" + str(expected_list) + "\nfound\n" + str(res_list))       

    if res_gen != expected_gen or res_list != expected_list:
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "h"
    arguments = [['a','b'], False]
    args = "a,b"
    expected_gen = "0{v_h_(a,b):dom_h(a,b)}1."
    expected_list = ['h(a,b,1609):-v_h_(a,b).', 'not_h(a,b,1609):- not v_h_(a,b).']
    res_gen, res_list = utilities.generate_generator(functor,args,arguments,0.2,1000)

    if res_gen != expected_gen:
        print("Error: expected\n" + str(expected_gen) + "\nfound\n" + str(res_gen))
    if res_list != expected_list:
        print("Error: expected\n" + str(expected_list) + "\nfound\n" + str(res_list))       

    if res_gen != expected_gen or res_list != expected_list:
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

    failed,passed = test_is_number()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_check_consistent_prob_fact()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_extract_atom_between_brackets()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_get_functor()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_generate_dom_fact()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    failed,passed = test_generate_generator()
    total = total + failed + passed
    n_function_tested = n_function_tested + 1

    print("Tested " + str(n_function_tested) + " functions")
    print("Executed " + str(total) + " tests")
    print("Passed: " + str(passed))
    print("Failed: " + str(failed))

if __name__ == "__main__":
    test_utilities()