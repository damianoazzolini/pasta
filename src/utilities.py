import sys
import re
from typing import Union

def is_number(n) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    return True

def test_is_number() -> Union[int,int]:
    # TODO
    return 0,0

# dummy consistency probabilistic fact
def check_consistent_prob_fact(line) -> Union[str,str]:
    line = line.split("::")
    # for example: line = ['0.5', 'f(1..3).']
    if len(line) != 2:
        print("Error in parsing: " + str(line))
        sys.exit()

    if not is_number(line[0]):
        print("Error: expected a float, found " + str(line[0]))
        sys.exit()
    
    return line

def test_check_consistent_prob_fact() -> Union[int,int]:
    # TODO
    return 0,0

# from f(12) returns 12, does some basic checks
# returns also True if range, false otherwise
def extract_atom_between_brackets(fact) -> Union[list, bool]:
    val = re.findall(r'\(([^()]+)\)',fact)
    if len(val) == 1:
        if ".." in val[0]:
            return val[0].split(".."), True
        else:
            return val[0].split(','), False
    else:
        print("Error in extracting arguments in " + fact)
        sys.exit()

def test_extract_atom_between_brackets() -> Union[int,int]:
    # TODO
    return 0,0

# from fa(32) returns fa
def get_functor(fact) -> str:
    r = ""
    i = 0
    print(fact)
    while fact[i] and fact[i] != '(':
        r = r + fact[i]
        i = i + 1
    return r

def test_get_functor() -> Union[int,int]:
    # TODO
    return 0,0

# generates the dom fact
# dom_f(1..3). from f(1..3).
def generate_dom_fact(functor,arguments) -> str:
    dom = "dom_" + functor + "("
    args = ""
    if arguments[1] is False: # single fact
        for a in arguments[0]:
            args = args + a + ","
        args = args[:-1] # remove last ,
        args = args + ")"
    else: # range
        # args = args + arguments[0][0] + ".." + arguments[0][1] + ")"
        args = "I)"

    if arguments[1] is False: # single fact
        dom_args = args
    else: # range
        dom_args = arguments[0][0] + ".." + arguments[0][1] + ")"
        
    dom = dom + dom_args + "."

    return dom

def test_generate_dom_fact() -> Union[int,int]:
    failed = 0
    passed = 0

    print("test_generate_dom_fact")

    functor = "f"
    arguments = [['1', '3'], True]
    expected = "dom_f(1..3)."
    dom = generate_dom_fact(functor,arguments)

    if dom != expected:
        print("Error: expected\n" + expected + "\nfound\n" + str(dom))
        failed = failed + 1
    else:
        passed = passed + 1

    functor = "g"
    arguments = [['a'], False]
    expected = "dom_g(a)."
    dom = generate_dom_fact(functor,arguments)

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