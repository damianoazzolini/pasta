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


def is_number(n : Union[int, float]) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    return True

# dummy consistency probabilistic fact
def check_consistent_prob_fact(line : str) -> Union[float,str]:
    line = line.split("::")
    # for example: line = ['0.5', 'f(1..3).']
    if len(line) != 2:
        print("Error in parsing: " + str(line))
        sys.exit()

    if not is_number(line[0]):
        print("Error: expected a float, found " + str(line[0]))
        sys.exit()

    prob = float(line[0])

    if prob > 1 or prob <= 0:
        print("Probabilities must be in the range ]0,1], found " + str(prob))
        sys.exit()
    
    return prob, line[1]

# from f(12) returns 12, does some basic checks
# returns also True if range, false otherwise
def extract_atom_between_brackets(fact : str) -> Union[list, bool]:
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
def get_functor(fact : str) -> str:
    r = ""
    i = 0
    while fact[i] and fact[i] != '(':
        r = r + fact[i]
        i = i + 1
    return r

# generates the dom fact
# dom_f(1..3). from f(1..3).
def generate_dom_fact(functor : str, arguments : list) -> Union[str,str]:
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

# TODO: modify test
def generate_generator(functor : str, args : str, arguments : list, prob : float, precision : int) -> Union[str,list]:
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
    log_prob_true = -int(math.log(prob)*(10**precision))
    log_prob_false = -int(math.log(1 - prob)*(10**precision))

    if number != 1: # clauses for range
        start = int(arguments[0][0])
        end = int(arguments[0][1])

        for i in range(start,end + 1):
            vt = "v_" + functor + "_(" + str(i) + ")"
            clause_true = functor + "(" + str(i) + "," + str(log_prob_true) + ")" + ":-" + vt + "."
            clause_false = "not_" + functor + "(" + str(i) + "," + str(log_prob_false) + ")" + ":- not " + vt + "."
            clauses.append(clause_true)
            clauses.append(clause_false)

        # add auxiliary clauses that wraps probabilities
        auxiliary_clause_true = functor + "(I) :- " + functor + "(I,_)."
        clauses.append(auxiliary_clause_true)

        auxiliary_clause_false = "not_" + functor + "(I) :- not_" + functor + "(I,_)."
        clauses.append(auxiliary_clause_false)

        # add show declaration
        show_declaration = "#show " + functor + "/2."
        clauses.append(show_declaration)

        show_declaration = "#show not_" + functor + "/2."
        clauses.append(show_declaration)

    else:
        clause_true = functor + "(" + args + "," + str(log_prob_true) + ")" + ":-" + vt + "."
        clause_false = "not_" + functor + "(" + args + "," + str(log_prob_false) + ")" + ":- not " + vt + "."
        clauses.append(clause_true)
        clauses.append(clause_false)

        auxiliary_clause_true = functor + "(" + args + ") :- " + functor + "(" + args + ",_)."
        clauses.append(auxiliary_clause_true)
        auxiliary_clause_false = "not_" + functor + "(" + args + ") :- not_" + functor + "(" + args + ",_)."
        clauses.append(auxiliary_clause_false)

        show_declaration = "#show " + functor + "/" + str(args.count(',') + 2) + "."
        clauses.append(show_declaration)
        show_declaration = "#show not_" + functor + "/" + str(args.count(',') + 2) + "."
        clauses.append(show_declaration)

    return generator, clauses

# ["bird(1,693)", "bird(2,693)", "bird(3,693)", "bird(4,693)", "nq"]
# returns 11213141, 693 + 693 + 693 + 693, True
# if nq in line -> returns True else False
# 11213141 means: 1 true, 2 true. 3 true, 4 true
# TODO: add test
def get_id_prob_world(line : str) -> Union[str,int]:
    line = line.split(' ')
    q = False
    id = ""
    prob = 0
    for term in line:
        if term == "q":
            q = True
        elif term == "nq":
            q = False
        else:
            term = term.split('(')
            if term[1].count(',') == 0: # arity original prob fact 0 (example: 0.2::a.)
                id = id + term[0]
                prob = prob + int(term[1][:-1])
            else:
                args = term[1][:-1].split(',')
                prob = prob + int(args[-1])
                id = id + term[0]
                for i in args[:-1]:
                    id = id + i
    
    return id, int(prob), q

def parse_command_line(args : str) -> Union[bool,bool,str,int,str]:
    verbose = False
    pedantic = False
    filename = ""
    precision = 3 # default value
    query = ""
    # for i in range(0,len(args)):
    i = 0
    while i < len(args):
        if args[i] == "--verbose" or args[i] == "-v":
            verbose = True
        elif args[i] == "--pedantic":
            verbose = True
            pedantic = True
        elif args[i].startswith("--precision=") or args[i].startswith("-p="):
            precision = int(args[i].split('=')[1])
        elif args[i] == "--help" or args[i] == "-h":
            print_help()
            sys.exit()
        elif args[i].startswith("--query=") or args[i].startswith("-q="):
            query = args[i].split("=")[1].replace("\"","")
        else:
            if i + 1 < len(args):
                filename = args[i+1]
                i = i + 1
        i = i + 1

    if filename == "":
        print("Missing filename")
        sys.exit()
    
    return verbose,pedantic,filename,precision,query

def print_help() -> None:
    print("paspsp: probabilistic answer set programming statistical probabilities")
    print("Compute lower and upper bound for a query in")
    print("a probabilistic answer set program")
    print("Example: paspsp ../../examples/bird_4.lp -q=\"fly(1)\"")
    print("Example programs: see example folder.")
    print("Issues: https://github.com/damianoazzolini/PaspStatsProb/issues")
    print("Available commands:")
    print("\t--query=,-q: specifies a query. Example: -q=\"fly(1)\".")
    print("\t--verbose,-v: verbose mode. Default: off.")
    print("\t--pedantic: pedantic mode (more verbose than --verbose). Default: off.")
    print("\t--precision=,-p=: set the required precision. Example: --precision=3. Default = 3.")
    print("\t--help,-h: print this help page")
