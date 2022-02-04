from curses.ascii import isupper
from typing import Union
import sys

import utils

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
    def generate_clauses_for_facts(term: str, probability: float, precision: int):
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
    def generate_clauses_for_conditionals(conditional : str) -> list:
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
    def generate_clauses_for_abducibles(line: str, n_abd : int) -> Union[str,str]:
        if len(line.split(' ')) != 2:
            utils.print_error_and_exit("Error in line " + line)
        
        if not line.endswith('.'):
            utils.print_error_and_exit("Missing final . in " + line)
        
        # TODO: add sanity checks for the atom: no variables and correct syntax
        line = line.split(' ')
        term = line[1][:-1]
        generator = '0{' + term + '}1.'
        t1 = "abd_" + term

        if "(" not in term:
            commas = 0
        else:
            commas = term.count(',') + 1

        new_fact_true = t1 + ':- ' + term + '.'

        new_fact_false = "not_" + t1 + ' :- not ' + term + '.'

        show_true = "#show " + t1.split('(')[0] + "/" + str(commas) + "."

        show_false = "#show not_" + t1.split('(')[0] + "/" + str(commas) + "."


        # show_true = "#show " + t1 + "/" + str(commas) + "."

        # show_false = "#show not_" + t1 + "/" + str(commas) + "."

        # used since in the program i insert a constraint that counts the abducibles
        abd_fact = "abd_fact(" + str(n_abd) + "):-" + term + "."

        return [generator, new_fact_true, new_fact_false, abd_fact, show_true, show_false], term

    # '''
    # Creates a prolog version of the line to identify the minimal set
    # of probabilistic and abducible facts
    # '''
    # @staticmethod
    # def to_prolog(line : str, n_ics : int, goal_index : int) -> str:
    #     res = []
    #     line = line.split(':-')
    #     ic_to_goal = False
    #     # process body
    #     if len(line) == 2:
    #         body = list(line[1])

    #         i = 0
    #         while i < len(body):
    #             if body[i] == '}':
    #                 while i < len(body) and body[i] != ',' and body[i] != '.':
    #                     body[i] = ' '
    #                     i = i + 1
    #             elif body[i] == '#':
    #                 # extract everything until :
    #                 while body[i] != ':':
    #                     body[i] = ' '
    #                     i = i + 1
    #                 body[i] = ' '
    #                 i = i + 1
    #                 temp = []
    #                 while i < len(body) and body[i] != '}':
    #                     temp.append(body[i])
    #                     body[i] = ' '
    #                     i = i + 1
    #                 body[i] = ' '  # }
    #                 # insert in the current position
    #                 body[i:i] = ['newGoalInserted', str(goal_index), ',']
    #                 i = i + 3

    #                 body_new = ''.join(temp).replace(' ', '')
    #                 # insert the new goal
    #                 goal = "newGoalInserted" + str(goal_index) + ':-' + body_new + '.'
    #                 goal_index = goal_index + 1
    #                 res.append(goal)
    #             elif body[i] == '=' or body[i] == '<' or body[i] == '>':
    #                 while i < len(body) and body[i] != ',' and body[i] != '.':
    #                     body[i] = ' '
    #                     i = i + 1
    #                 body[i] = ' '
    #                 i = i + 1
    #             else:
    #                 i = i + 1
    #         # second pass to remove other symbols
    #         # print(body)
    #         i = 0
    #         while i < len(body):
    #             if body[i] == '=' or body[i] == '<' or body[i] == '>' or body[i] == '+' or body[i] == '-' or body[i] == '*':
    #                 i_forward = i
    #                 while i_forward < len(body) and body[i_forward] != ',' and body[i_forward] != '.':
    #                     body[i_forward] = ' '
    #                     i_forward = i_forward + 1
    #                 # body[i_forward] = ' '
    #                 i_backward = i
    #                 while i_backward > 0 and body[i_backward] != ',' and body[i_backward] != '.':
    #                     body[i_backward] = ' '
    #                     i_backward = i_backward - 1
    #                 body[i_backward] = ' '
    #                 i = i + 1
    #             elif len(body[i]) == 1 and isupper(body[i]):
    #                 if i > 0 and body[i-1] == ' ':
    #                     while i < len(body) and body[i] != ' ' and body[i] != ',' and body[i] != '.':
    #                         body[i] = ' '
    #                         i = i + 1
    #                 else:
    #                     i = i + 1
    #             else:
    #                 i = i + 1
    #     else:
    #         body = None

    #     if body is not None:
    #         body = ''.join(body).replace(' ', '')
    #         # case where the goal is the last to be inserted
    #         if body.endswith(','):
    #             body = body[:-1] + '.'
    #         if not body.endswith('.'):
    #             body = body + "."

    #     # process head
    #     head = line[0]
    #     if ';' in head:
    #         # disjunctive head
    #         head = head.split(';')
    #         for h in head:
    #             res.append(h + (':-' + body if body is not None else ''))
    #     elif '{' in head:
    #         #head aggregate
    #         head = list(head)
    #         i = 0
    #         while i < len(head) and head[i] != '{':
    #             head[i] = ' '
    #             i = i + 1
    #         head[i] = ' '
    #         i = i + 1

    #         while i < len(head) and head[i] != '}':
    #             i = i + 1
    #         head[i] = ' '
    #         i = i + 1

    #         while i < len(head) and head[i] != ':' and head[i] != '.':  # . or :- terminator
    #             head[i] = ' '
    #             i = i + 1

    #         head = ''.join(head).replace(' ', '')
    #         res.append(head + (':-' + body if body is not None else ''))
    #     elif head == '':
    #         # integrity constraint
    #         head = "newGoalFromICInserted" + str(n_ics)
    #         ic_to_goal = True
    #         n_ics = n_ics + 1
    #         res.append(head + (':-' + body if body is not None else ''))

    #     else:
    #         res.append(head + (':-' + body if body is not None else ''))
    
    #     return res, n_ics, goal_index, ic_to_goal

