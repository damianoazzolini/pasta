from sympy import *
# from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import shgo

from itertools import combinations
import sys

from typing import Any

from .utils import is_number

def simplify_chunk(eq: str, chunk_size : int = 1_000_000) -> str:
    return eq
    # simplify chunk_size sums at the time
    span = int(eq.count('+') / chunk_size)
    if span == 0:
        return str(sympify(eq))
    
    elements = eq.split('+')
    new_eq_list = ["+".join(elements[i:i+span]) for i in range(0, len(elements), span)]
    
    
    new_eq = ""
    
    for el in new_eq_list:
        to_append = str(sympify(el))
        # print(f"to append: {to_append}")
        if to_append.startswith('-') or len(new_eq) == 0:
            new_eq = new_eq + to_append
        else:
            new_eq = new_eq + '+' + to_append
    
    if new_eq.endswith('+'):
        new_eq = new_eq[:-1]

    return sympify(new_eq)
    

class Problem:
    def __init__(
        self,
        function_to_opt : str,
        symbolic_variables : 'list[str]'
        ) -> None:
        self.function_to_opt = function_to_opt
        self.symbolic_variables = symbolic_variables


    def eval_fn(self, x : 'list[float]', other_to_eval : str = ""):
        '''
        Evaluates the constraint function
        '''
        if other_to_eval == "":
            s = str(self.function_to_opt)
        else:
            s = str(other_to_eval)
        
        for idx, val in enumerate(self.symbolic_variables):
            s = s.replace(val, str(x[idx]))
        # # return sympify(s)
        # return simplify_chunk(s)
        return eval(s)


    def jac_fn(self, x: 'list[float]'):
        '''
        Jacobian function
        '''
        j = []
        s = self.function_to_opt
        for symbolic_var in self.symbolic_variables:
            d = diff(s,symbolic_var)
            j.append(self.eval_fn(x,str(d)))
        return j


def compute_optimal_probability(
        initial_equation : 'str | tuple[str,str]',
        optimizable_facts: 'dict[str, tuple[float, float]]',
        probability_threshold : float, # p(q) > probability_threshold
        epsilon : float = -1,
        method : str = "SLSQP",
        chunk_size : int = 1_000_000,
        credal_facts : bool = False
    ):
    '''
    Compute the optimal value to associate to probabilistic facts.
    By now, I only allow constraints on the probability value of the
    query.
    '''

    simplified_equation_lp = ""
    simplified_equation_up = ""
    simplified_equation = ""

    if isinstance(initial_equation, tuple): # credal facts
        simplified_equation_lp = simplify_chunk(initial_equation[0] + f"- {probability_threshold}", chunk_size)
        simplified_equation_up = simplify_chunk(initial_equation[1] + f"- {probability_threshold}", chunk_size)
        for eq_s in [simplified_equation_lp,simplified_equation_up]:
            print(f"number of sums: {str(eq_s).count('+')}")
            print(f"number of prods: {str(eq_s).count('*')}")
    else:
        simplified_equation = simplify_chunk(initial_equation + f"- {probability_threshold}", chunk_size)
        print(f"number of sums: {simplified_equation.count('+')}")
        print(f"number of prods: {simplified_equation.count('*')}")
    # 1.1: if is a number, return it
    # if is_number(str(simplified_equation)):
    #     return simplified_equation
    opt_facts_names = list(optimizable_facts.keys())
    
    # the target is to minimize the sum of the prob of the optimizable facts
    if not credal_facts:
        target_equation = "+".join(opt_facts_names)
        problem_to_solve = Problem(target_equation, opt_facts_names)

    # 2: generate the bounds
    bounds : 'list[tuple[float,float]]' = []
    bounds_constraints_cobyla : 'list[str]' = []

    if method == "SLSQP":
        for bound in optimizable_facts.values():
            bounds.append((bound[0],bound[1]))
    else:
        for k, v in optimizable_facts.items():
            bounds_constraints_cobyla.append(f"{k} - {v[0]}")
            bounds_constraints_cobyla.append(f"{v[1]} - {k}")
    
    initial_guesses : 'list[float]' = [0.5]*len(optimizable_facts)
    constraints : 'list[dict[str,Any]]' = []
    
    if isinstance(initial_equation, tuple): # credal facts
        pass
    else:
        query_constraint = Problem(
            simplified_equation,
            opt_facts_names
        )
        constraints.append({
            'type' : 'ineq',
            'fun' : query_constraint.eval_fn,
            'jac' : query_constraint.jac_fn    
        })
    
    for el in bounds_constraints_cobyla:
        current_constraint = Problem(el, opt_facts_names)
        constraints.append({
            'type' : 'ineq',
            'fun' : current_constraint.eval_fn,
            'jac' : current_constraint.jac_fn    
        })

    # 2.1: if epsilon != -1, impose a constraint x_1 - x_j < epsilon
    # for each pair of optimizable facts
    if epsilon > 0:
        # for pair in combinations(enumerate(optimizable_facts.keys()),2):
        for pair in combinations(range(len(optimizable_facts)),2):
            # fun01 = lambda x : x[pair[0]] - x[pair[1]]
            # fun10 = lambda x : x[pair[1]] - x[pair[0]]
            # constraints.append(NonlinearConstraint(fun01, -1, epsilon))
            # constraints.append(NonlinearConstraint(fun10, -1, epsilon))
            current_constraint_01 = Problem(
                f"-({opt_facts_names[pair[0]]} - {opt_facts_names[pair[1]]} - {epsilon})",
                list(optimizable_facts.keys())
            )
            current_constraint_10 = Problem(
                f"-({opt_facts_names[pair[1]]} - {opt_facts_names[pair[0]]} - {epsilon})", 
                list(optimizable_facts.keys())
            )            
            constraints.append({
                'type' : 'ineq',
                'fun' : current_constraint_01.eval_fn,
                'jac' : current_constraint_01.jac_fn    
            })
            constraints.append({
                'type' : 'ineq',
                'fun' : current_constraint_10.eval_fn,
                'jac' : current_constraint_10.jac_fn    
            })
    
    res = []

    if method == "SLSQP":
        if credal_facts:
            for idx, prob_eq in enumerate([simplified_equation_lp,simplified_equation_up]):
                if idx == 0:
                    problem_to_solve = Problem(prob_eq, opt_facts_names)
                else:
                    problem_to_solve = Problem(f"-({prob_eq})", opt_facts_names)

                res_v = minimize(
                    problem_to_solve.eval_fn,
                    initial_guesses,
                    bounds=bounds,
                    jac=problem_to_solve.jac_fn,
                    method=method
                )

                res.append(res_v)
        else: # optimizable task
            res = minimize(
                problem_to_solve.eval_fn,
                initial_guesses,
                bounds=bounds,
                jac=problem_to_solve.jac_fn,
                method=method,
                constraints=constraints
            )
    else:
        if credal_facts:
            for idx, prob_eq in enumerate([simplified_equation_lp,simplified_equation_up]):
                if idx == 0:
                    problem_to_solve = Problem(prob_eq, opt_facts_names)
                else:
                    problem_to_solve = Problem(f"-({prob_eq})", opt_facts_names)
                
                res_v = minimize(
                    problem_to_solve.eval_fn,
                    initial_guesses,
                    method=method,
                    constraints=constraints
                )
                res.append(res_v)


        else: # optimizable task
            res = minimize(
                problem_to_solve.eval_fn,
                initial_guesses,
                method=method,
                constraints=constraints
            )

    return res
