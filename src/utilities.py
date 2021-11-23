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
    
    return float(line[0]), line[1]

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

# generates the model_query and model_not_query clauses
# Clauses model_query and model_not_query structure:
# model_query(NPF1T,SPF1T,NPF1F,SPF1F,...)
# where:
#   - NPF1T: number of probabilistic fact 1 true
#   - SPF1T: sum of the probabilities of probabilistic fact 1 true
#   - NPF1F: number of probabilistic fact 1 false
#   - NPF1F: sum of the probabilities of probabilistic fact 1 false
# this for every different possible probability for probabilistic
# facts: 4*n arguments
# the tuple (NPF1T NPF1F NPF2T ... ) identifies the world
# TODO: here, i generate two clauses with the same body for
# model_query and model_not_query: wrap it in another predicate
def generate_model_clause(functors_list : list, query : str) -> Union[str,str]:
    clauses_model = [] # clauses for model query and model not query
    prototype_arguments_model_clauses = ""
    args_counter = 0
    clause_count_true = ""

    prob_fact = False

    for functor in functors_list:
        if "(" in functor: # probabilistic fact
            prob_fact = True
            functor = functor[:-1] + ","
        else:
            # to reuse some operations
            functor = functor + "("

        # clause count true
        clause_count_true = "I" + str(args_counter) + " = #count{ X : " + functor + "X) }"
        clauses_model.append(clause_count_true)

        prototype_arguments_model_clauses = prototype_arguments_model_clauses + "I" + str(args_counter) + ","
        args_counter = args_counter + 1

        # clause sum true
        if prob_fact:
            clause_sum_true = "I" + str(args_counter) + " = #sum{ X : " + functor + "X) }"
        else:
            clause_sum_true = "I" + str(args_counter) + " = #sum{ Y,X : dom_" + functor + "X), " + functor + "X,Y) }"
        
        clauses_model.append(clause_sum_true)
        
        prototype_arguments_model_clauses = prototype_arguments_model_clauses + "I" + str(args_counter) + ","
        args_counter = args_counter + 1

        # clause count false
        clause_count_false = "I" + str(args_counter) + " = #count{ X : not_" + functor + "X) }"
        clauses_model.append(clause_count_false)

        prototype_arguments_model_clauses = prototype_arguments_model_clauses + "I" + str(args_counter) + ","
        args_counter = args_counter + 1

        # clause sum false
        if prob_fact:
            clause_sum_false = "I" + str(args_counter) + " = #sum{ X : not_" + functor + "X) }"
        else:
            clause_sum_false = "I" + str(args_counter) + " = #sum{ Y,X : dom_" + functor + "X), not_" + functor + "X,Y) }"
        
        clauses_model.append(clause_sum_false)
        prototype_arguments_model_clauses = prototype_arguments_model_clauses + "I" + str(args_counter) + ","
        args_counter = args_counter + 1

    prototype_arguments_model_clauses = prototype_arguments_model_clauses[:-1] # remove the last ,
    model_query_clause = "model_query(" + prototype_arguments_model_clauses + "):- "

    model_not_query_clause = "model_not_query(" + prototype_arguments_model_clauses + "):- "

    for el in clauses_model:
        model_query_clause = model_query_clause + el + ", "
        model_not_query_clause = model_not_query_clause + el + ", "

    model_query_clause = model_query_clause + query + ".\n" + "#show model_query/" + str(args_counter) + "."
    model_not_query_clause = model_not_query_clause + "not " + query + ".\n" + "#show model_not_query/" + str(args_counter) + "."

    return model_query_clause, model_not_query_clause

# TODO: modify test
def generate_generator(functor : str, args : str, arguments : list, prob : float, precision : int) -> Union[str,list,list]:
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
    log_prob_true = -int(math.log(prob)*precision)
    log_prob_false = -int(math.log(1 - prob)*precision)

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

        # add show declaration for auxiliary clauses
        show_declaration = "#show " + functor + "/1."
        clauses.append(show_declaration)

        show_declaration = "#show not_" + functor + "/1."
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

        show_declaration = "#show " + functor + "/" + str(args.count(',') + 1) + "."
        clauses.append(show_declaration)
        show_declaration = "#show not_" + functor + "/" + str(args.count(',') + 1) + "."
        clauses.append(show_declaration)

        # update the functor, since it is used in the function generate_model_clause
        functor = functor + "(" + args + ")" 

    return generator, clauses, functor

# gets a line 1,693,1,693,2,1832,0,0
# return 1120, 693 + 693 + 1832 + 0
# TODO: add test
def get_id_prob_world(line : str) -> Union[str,int]:
    w = line.split(',')[::2] # only even positions
    p = line.split(',')[1::2] # only even positions
    return ''.join(w), sum(int(x) for x in p)