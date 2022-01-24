from typing import Counter, Union
import sys
'''
Class defining a generator of an ASP program
'''

class Generator:
    def __init__(self):
        pass

    # generates the dom fact
    # Improve: from f(1..3) to f(1) f(2) f(3)
    # dom_f(1..3). from f(1..3).
    # dom_a from a
    @staticmethod
    def generate_clauses_from_facts(term: str, probability: float, precision: int):
        generator_term = '0{' + term + '}1.'

        if "(" not in term:
            t1 = term + "("
            commas = 0
        else:
            # compound
            t1 = term[:-1] + ","
            commas = term.count(',') + 1

        new_fact_true = t1 + str(int(probability * (10**precision))) + ') :- ' + term + '.'

        new_fact_false = "not_" + t1 + str(int((1 - probability) * (10**precision))) + ') :- not ' + term + '.'

        show_true = "#show " + term.split('(')[0] + "/" + str(commas + 1) + "."

        show_false = "#show not_" + term.split('(')[0] + "/" + str(commas + 1) + "."

        return [generator_term,new_fact_true,new_fact_false,show_true,show_false]

    @staticmethod
    def extract_vars(term: str) -> list:
        term = term.replace('(', ',').replace(')', ',')
        term = term.split(',')
        return [var for var in term if (len(var) > 0 and var[0].isupper())]

    @staticmethod
    def expand_conditional(conditional : str) -> list:
        if "|" not in conditional:
            sys.exit("Syntax error in conditional: " + conditional)
        if "[" not in conditional or "]" not in conditional:
            sys.exit("Missing range in conditional: " + conditional)
        if not conditional.endswith("."):
            sys.exit("Missing final . in " + conditional)
        
        conditional = conditional[:-1]
        i = 1
        par_count = 1
        while (par_count > 0) and i < len(conditional):
            if conditional[i] == '(':
                par_count = par_count + 1
            elif conditional[i] == ')':
                par_count = par_count - 1
            i = i + 1

        if i == len(conditional):
            sys.exit("Syntax error in conditional: " + conditional)

        cond, range = conditional[1:i-1], conditional[i:]

        cond = cond.split("|")
        if len(cond) != 2:
            sys.exit("Too many | in " + conditional)
        
        vars = Generator.extract_vars(cond[0])

        disjunct = cond[0] + " ; not_" + cond[0] + " :- " + cond[1] + "."
        # here I consider only one term in the left part
        # f(a,b) | ... not f(a,b), f(b,c) | ...
        constr = ":- #count{" + ','.join(vars) + ":" + cond[1] + "} = H, #count{" + ','.join(vars) + ":" + cond[0] + "," + cond[1] + "} = FH"

        range = range.split(",")
        if len(range) != 2:
            sys.exit("Unbalanced range in conditional: " + conditional)
        lower = range[0][1:]
        upper = range[1][:-1]

        if float(lower) == 0 and float(upper) != 0:
            ub = int(float(lower) * 100)
            cu = constr +  ", " + "100 * FH > " + str(ub) + "*H."
            cl = ""
        elif float(upper) == 1 and float(lower) != 0:
            lb = int(float(lower) * 100)
            cl = constr + ", " + "100 * FH < " + str(lb) + "*H."
            cu = ""
        elif float(upper) != 1 and float(lower) != 0:
            ub = int(float(lower) * 100)
            cu = constr + ", " + "100 * FH > " + str(ub) + "*H."
            lb = int(float(lower) * 100)
            cl = constr + ", " + "100 * FH < " + str(lb) + "*H."

        return [disjunct,cu,cl]

    @staticmethod
    def expand_abducible(line: str) -> Union[str,str]:
        if len(line.split(' ')) != 2:
            sys.exit("Error in line " + line)
        
        if not line.endswith('.'):
            sys.exit("Missing final . in " + line)
        
        # TODO: add sanity checks for the atom: no variables and correct syntax
        line = line.split(' ')
        return '0{' + line[1][:-1] + '}1.', line[1][:-1]