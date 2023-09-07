from gekko import GEKKO
from sympy import simplify, expand

import re

from .utils import is_number, print_error_and_exit

def reduce_pasp_up(
    equation : str,
    reducible_facts : dict[str,float],
    probability_threshold : float,
    pedantic : bool = True
    ) -> tuple[bool,list[int],float]:
    '''
    Solution of the reducible task considering the UP.
    Returns:
    - bool: solution found
    - list[int]: list of 0-1 identifying whether the ith fact has been selected
    - float: computed probability
    '''
    m = GEKKO(remote=False)  # Initialize gekko
    m.options.SOLVER = 1  # APOPT is an MINLP solver

    # optional solver settings with APOPT
    m.solver_options = ['minlp_maximum_iterations 5000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 100', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 500', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.0005', \
                        # covergence tolerance
                        'minlp_gap_tol 0.00001']

    # Initialize variables
    # a = m.Var(value=1,lb=1,ub=5)
    # b = m.Var(value=5,lb=1,ub=5)
    # Integer constraints for x3 and x4
    variables = {}
    i = 0
    for name, prob in reducible_facts.items():
        # prob should always be 1
        variables[name] = [prob, m.Var(value=1, lb=0, ub=1, integer=True)]
        # variables[f"x{i}"] = [prob, m.Var(value=0, lb=0, ub=1, integer=True)]
        i += 1
    # variables = {
    #     'a': [0.4, m.Var(value=0, lb=0, ub=1, integer=True)],
    #     'b': [0.7, m.Var(value=0, lb=0, ub=1, integer=True)]
    # }

    # symp = "0.5*a"

    print(equation)
    
    eq_to_handle = str(simplify(equation)).replace(' ', '')
    
    # this to remove parentheses
    eq_to_handle = str(expand(eq_to_handle)).replace(' ', '')
    
    if pedantic:
        print("Simplified equation")
        print(eq_to_handle)
        n_plus = eq_to_handle.count('+')
        n_minus = eq_to_handle.count('-')
        n_prod = eq_to_handle.count('*')
        print(f"Number of operations: {n_plus + n_minus + n_prod}")
        print(f"- sums (+): {n_plus}")
        print(f"- subs (-): {n_minus}")
        print(f"- n_prod (*): {n_prod}")
        print(f"len: {len(eq_to_handle)}")
    
    splitted_eq_ = re.split('(\W)', eq_to_handle) # split words
    
    # cleanup the equation, since in the previous step the 
    # numbers are splitted: ['0','.','12']
    i = 0
    splitted_eq : 'list[str]' = []
    while i < len(splitted_eq_):
        if len(splitted_eq_[i]) > 0:
            if i+2 < len(splitted_eq_) and is_number(splitted_eq_[i]) and splitted_eq_[i+1] == '.' and is_number(splitted_eq_[i+2]):
                splitted_eq.append(splitted_eq_[i]+splitted_eq_[i+1]+splitted_eq_[i+2])
                i = i+3
            else:
                splitted_eq.append(splitted_eq_[i])
                i = i + 1
        else:
            i = i + 1
    
    prev = None
    probability_equation_as_string = ''
    probability_equation = None
    p_eq_list = []
    in_bracket = False

    # all this since too long eqautions are not supported by GEKKO
    # so instead of one big equation I pass k small equations
    # print(splitted_eq)
    for el in splitted_eq:
        # print(el)
        if el == '+' or el == '-' or el == '*':
            prev = el
            if el != "*" and not in_bracket and probability_equation is not None:
                p_eq_list.append(probability_equation)
            # elif el == ')' and in_bracket and probability_equation is not None:
            #     p_eq_list.append(probability_equation)
            #     in_bracket = False
                probability_equation = None
        elif el == '(':
            in_bracket = True
        elif el == ')':
            in_bracket = False
        else: # to avoid el = ''
            if is_number(el):
                v = str(el)
                to_insert = float(el)
            else:
                v = str(variables[el][0]) + f'*{el}'
                to_insert = variables[el][1]
            probability_equation_as_string += str(prev) + v if prev is not None else v
            if prev == '+' or prev is None:
                if probability_equation is not None:
                    probability_equation += to_insert
                else:
                    probability_equation = to_insert
            elif prev == '-':
                if probability_equation is not None:
                    probability_equation -= to_insert
                else:
                    probability_equation = -to_insert
            elif prev == '*':
                probability_equation *= to_insert

            prev = None

    # add the last one
    if probability_equation is not None:
        p_eq_list.append(probability_equation)
    
    if pedantic:
        print("GEKKO APOPT Equation")
        print(p_eq_list)
    
    # constraint
    m.Equation(m.sum(p_eq_list) >= probability_threshold)

    # Objective
    m.Obj(sum(v[1] for v in variables.values()))  # Objective
    found = False
    try:
        m.solve(disp=False)  # Solve
        found = True
    except Exception as e:
        if "@error: Solution Not Found" in str(e):
            print("Solution not found")
        elif "@error: Max Equation Length" in str(e):
            print("Query equation too long")
            print(e)
        else:
            print(e)
            print_error_and_exit("An error occurred within GEKKO.")
        print(e)

    
    if found and pedantic:
        print('Results')
        for k, v in variables.items():
            print(f'{k}: ' + str(v[1].value))

        # m.options fields: https://gekko.readthedocs.io/en/latest/global.html
        print('Objective: ' + str(m.options.objfcnval))

    selected : 'list[str]' = []
    for k, v in variables.items():
        selected.append(str(v[1].value))
        probability_equation_as_string = probability_equation_as_string.replace(k,str(v[1].value[0]))
    
    return found, selected, eval(probability_equation_as_string)
    