import sys
import re
from typing import Union

def is_number(n) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    return True

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

# from fa(32) returns fa
def get_functor(fact) -> str:
    r = ""
    i = 0
    print(fact)
    while fact[i] and fact[i] != '(':
        r = r + fact[i]
        i = i + 1
    return r

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