import sys
import re
import math
from typing import Union

# TODO: write test for this
def endline_content(char1) -> bool:
    return char1 == '\n' or char1 == '\r\n' or char1 == ' '

# TODO: write test for this
def endline_comment(char1) -> bool:
    return char1 == '\n' or char1 == '\r\n'


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
    while fact[i] and fact[i] != '(':
        r = r + fact[i]
        i = i + 1
    return r

# generates the dom fact
# dom_f(1..3). from f(1..3).
def generate_dom_fact(functor,arguments) -> Union[str,str]:
    dom = "dom_" + functor + "("
    args = ""
    if arguments[1] is False: # single fact
        for a in arguments[0]:
            args = args + a + ","
        args = args[:-1] # remove last ,
    else: # range
        # args = args + arguments[0][0] + ".." + arguments[0][1] + ")"
        args = "I"

    if arguments[1] is False: # single fact
        dom_args = args
    else: # range
        dom_args = arguments[0][0] + ".." + arguments[0][1]
        
    dom = dom + dom_args + ")."

    return dom,args

def generate_generator(functor,args,arguments,prob,precision) -> Union[str,list]:
    vt = "v_" + functor + "_(" + args + ")"
    generator = ""
    generator = generator + "0{"
    if arguments[1] is False:
        number = 1
    else:
        number = int(arguments[0][1]) - int(arguments[0][0]) + 1
    
    generator = generator + vt + ":dom_" + functor + "(" + args + ")}" + str(number) + "."

    # generate the two clauses
    clauses = []
    log_prob = -int(math.log(float(prob))*precision)

    if number != 1: # clauses for range
        start = int(arguments[0][0])
        end = int(arguments[0][1])

        for i in range(start,end + 1):
            vt = "v_" + functor + "_(" + str(i) + ")"
            clause_true = functor + "(" + str(i) + "," + str(log_prob) + ")" + ":-" + vt + "."
            clause_false = "not_" + functor + "(" + str(i) + "," + str(log_prob) + ")" + ":- not " + vt + "."
            clauses.append(clause_true)
            clauses.append(clause_false)

        # add show declaration
        auxiliary_clause_true = functor + "(I) :- " + functor + "(I,_)."
        clauses.append(auxiliary_clause_true)
        auxiliary_clause_false = "not_" + functor + "(I) :- not_" + functor + "(I,_)."
        clauses.append(auxiliary_clause_false)

        show_declaration = "#show " + functor + "/1."
        clauses.append(show_declaration)
        show_declaration = "#show not_" + functor + "/1."
        clauses.append(show_declaration)

    else:
        clause_true = functor + "(" + args + "," + str(log_prob) + ")" + ":-" + vt + "."
        clause_false = "not_" + functor + "(" + args + "," + str(log_prob) + ")" + ":- not " + vt + "."
        clauses.append(clause_true)
        clauses.append(clause_false)

        auxiliary_clause_true = functor + "(" + args + ") :- " + functor + "(" + args + ",_)."
        clauses.append(auxiliary_clause_true)
        auxiliary_clause_false = "not_" + functor + "(" + args + ") :- not_" + functor + "(" + args + ",_)."
        clauses.append(auxiliary_clause_false)

        show_declaration = "#show " + functor + "/" + str(args.count(',') + 1)
        clauses.append(show_declaration)
        show_declaration = "#show not_" + functor + "/" + str(args.count(',') + 1)
        clauses.append(show_declaration)

    return generator,clauses