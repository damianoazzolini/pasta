import sys

from utils import print_error_and_exit

'''
Class defining a generator of an ASP program
'''

class Generator:
    def __init__(self):
        pass


    @staticmethod
    def generate_clauses_for_facts(term : str) -> 'list[str]':
        generator_term = '0{' + term + '}1.'
        fact_false = f"not_{term}:- not {term}."
        show_true = f"#show.\n#show {term}:{term}."
        show_false = f"#show not_{term}:not_{term}."
        return [generator_term, fact_false, show_true, show_false]


    @staticmethod
    def extract_vars(term: str) -> 'list[str]':
        term = term.replace('(', ',').replace(')', ',')
        term_list = term.split(',')
        return [var for var in term_list if (len(var) > 0 and var[0].isupper())]


    @staticmethod
    def generate_clauses_for_conditionals(conditional : str) -> 'list[str]':
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

        # disjunctive rules are not ok, I need to use choice rules
        # disjunct = cond[0] + " ; not_" + cond[0] + " :- " + cond[1] + "."
        disjunct = f"0{{ {cond[0]} }}1 :- {cond[1]}."
        # here I consider only one term in the left part
        # f(a,b) | ... not f(a,b), f(b,c) | ...
        constr = ":- #count{" + ','.join(vars) + ":" + cond[1] + "} = H, #count{" + ','.join(vars) + ":" + cond[0] + "," + cond[1] + "} = FH"

        range = range.split(",")
        if len(range) != 2:
            sys.exit("Unbalanced range in conditional: " + conditional)
        lower = range[0][1:]
        upper = range[1][:-1]

        cu = ""
        cl = ""

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
    def generate_clauses_for_abducibles(line: str, n_abd: int) -> 'tuple[list[str], str]':
        if len(line.split(' ')) != 2:
            print_error_and_exit("Error in line " + line)
        
        if not line.endswith('.'):
            print_error_and_exit("Missing final . in " + line)
        
        # TODO: add sanity checks for the atom: no variables and correct syntax
        line_list = line.split(' ')
        term = line_list[1][:-1]
        generator = '0{' + term + '}1.'
        t1 = "abd_" + term

        new_fact_true = t1 + ':- ' + term + '.'
        new_fact_false = "not_" + t1 + ' :- not ' + term + '.'
        show_true = f"#show.\n#show {t1}:{t1}."
        show_false = f"#show not_{t1}:not_{t1}."
        abd_fact = "abd_fact(" + str(n_abd) + "):-" + term + "."

        return [generator, new_fact_true, new_fact_false, abd_fact, show_true, show_false], term
