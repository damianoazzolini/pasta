import math
import random
import copy

from .utils import print_error_and_exit
from . import continuous_cdfs


class ComparisonPredicate:
    '''
    Class representing an interval.
    '''

    def __init__(self, comparison_type: str, bound1: float, bound2: float = -math.inf) -> None:
        # comparison_type in ["above","below","between","outside"]
        self.comparison_type: str = comparison_type
        self.bound1: float = bound1
        self.bound2: float = bound2

    def __str__(self) -> str:
        if self.bound2 == -math.inf:
            return f"{self.comparison_type}: [{self.bound1}]"
        else:
            return f"{self.comparison_type}: [{self.bound1}, {self.bound2}]"


    def __repr__(self) -> str:
        return self.__str__()


class Generator:
    '''
    Class defining a generator of an answer set program
    '''
    ad_count : int = 0 # static variable

    def __init__(self):
        pass

    @staticmethod
    def generate_clauses_for_dt(term: str, type: str, naive : bool = False) -> 'list[str]':
        t1 = ""
        if type == "utility":
            t1 = "utility_" + term # really needed?
            generator = ""
        else:
            generator = '{' + term + '}.'
            t1 = "decision_" + term

        # TODO: add utility(f,10):- f. to allow aggregates over utilities
        new_fact_true = t1 + ':- ' + term + '.'
        new_fact_false = "not_" + t1 + ' :- not ' + term + '.'
        # do not show when naive solver is selected
        if (type == "utility" or type == "decision") and naive is False:
            show_true = f"#show.\n#show {t1}:{t1}."
            show_false = f"#show not_{t1}:not_{t1}."
        else:
            show_true = ""
            show_false = ""
            
        return [generator, new_fact_true, new_fact_false, show_true, show_false]


    @staticmethod
    def generate_clauses_for_facts(term : str, approx : bool = False, lpmln : bool = False) -> 'list[str]':
        if approx or lpmln:
            if approx:
                generator_term = f'#external {term}.'
            else:
                generator_term = '{' + term + '}.'
            fact_false = ""
            show_true = ""
            show_false = ""
        else:
            generator_term = '0{' + term + '}1.'
            fact_false = f"not_{term}:- not {term}."
            show_true = f"#show.\n#show {term}:{term}."
            show_false = f"#show not_{term}:not_{term}."
        return [generator_term, fact_false, show_true, show_false]


    @staticmethod
    def generate_clauses_for_facts_for_asp_solver(index: int, term: str, probability: float) -> 'list[str]':
        lp = math.ceil(math.log(probability)*1000)
        lnp = math.ceil(math.log(1 - probability)*1000)
        term_p = f"__a({index},{lp})"
        term_np = f"not__a({index},{lnp})"

        generator_term = "{" + term_p + "}." + f"\n{term_np}:- not {term_p}."
        wrap_fact_true = f"{term} :- {term_p}.\n__a({index}):- {term_p}."
        fact_false = f"not_{term}:- not {term_p}."
        show_true = f"#show {term}:{term}."
        show_false = f"#show not_{term}:not_{term}."
        maximize_statement = "#maximize{X,Y:__a(Y,X); X,Y:not__a(Y,X)}."

        return [generator_term, wrap_fact_true, fact_false, show_true, show_false, maximize_statement]


    @staticmethod
    def extract_vars(term: str) -> 'list[str]':
        term = term.replace('(', ',').replace(')', ',')
        term_list = term.split(',')
        return [var for var in term_list if (len(var) > 0 and var[0].isupper())]


    @staticmethod
    def generate_clauses_for_conditionals(conditional : str) -> 'list[str]':
        if "|" not in conditional:
            print_error_and_exit(f"Syntax error in conditional {conditional}")
        if "[" not in conditional or "]" not in conditional:
            print_error_and_exit(f"Missing range in conditional {conditional}")
        if not conditional.endswith("."):
            print_error_and_exit(f"Missing final . in {conditional}")

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
            print_error_and_exit(f"Syntax error in conditional {conditional}")

        cond, prob_range = conditional[1:i-1], conditional[i:]

        cond = cond.split("|")
        if len(cond) != 2:
            print_error_and_exit(f"Too many | in {conditional}")

        variables = Generator.extract_vars(cond[0])
        body_atoms : "list[str]" = []
        init_pos = 0
        body = cond[1]
        for i in range(0,len(body)):
            if body[i] == ')':
                body_atoms.append(body[init_pos:i+1])
                init_pos = i + 1

        for el in body_atoms:
            variables = variables + Generator.extract_vars(el)
        # remove duplicates
        variables = list(set(variables)) 

        # disjunctive rules are not ok, I need to use choice rules
        # disjunct = cond[0] + " ; not_" + cond[0] + " :- " + cond[1] + "."
        # disjunct = f"0{{ {cond[0]} }}1 :- {cond[1]}."
        disjunct = f"{cond[0]} ; not_{cond[0]} :- {cond[1]}."
        # here I consider only one term in the left part
        # f(a,b) | ... not f(a,b), f(b,c) | ...
        constr = ":- #count{" + ','.join(variables) + ":" + cond[1] + "} = H, #count{"+\
            ','.join(variables) + ":" + cond[0] + "," + cond[1] + "} = FH"

        prob_range = prob_range.split(",")
        if len(prob_range) != 2:
            print_error_and_exit("Unbalanced range in conditional: " + conditional)
        lower = float(prob_range[0][1:])
        upper = float(prob_range[1][:-1])

        if lower > upper:
            print_error_and_exit(f"LB cannot be greater than UB in {conditional}")

        cu = ""
        cl = ""

        if float(upper) != 1:
            ub = int(upper * 100)
            cu = f"{constr}, 100*FH > {ub}*H."
        if float(lower) != 0:
            lb = int(lower * 100)
            cl = f"{constr}, 100*FH < {lb}*H."

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
    

    @staticmethod
    def generate_xor_constraint(n_vars : int):
        
        def flip():
            return random.randint(0,1) == 1

        constr = ":- #count{"

        for i in range(0, n_vars):
            if flip():
                # constr = constr + f"1,bird({i}) : bird({i});"
                constr = constr + f"1,__a({i}) : __a({i});"
        if constr.endswith("{"):
            # no constraints were added
            return ""
        # random even/odd constraint
        parity = 0 if flip() else 1
        return constr[:-1] + "} = N, N\\2 = " + str(parity) + "."


    @staticmethod
    def generate_facts_from_disjunction(line : str) -> 'tuple[list[str],list[str]]':
        new_facts : list[str] = []
        new_clauses : list[str] = []
        prob_list : 'list[float]' = []
        name : str = f"__aux_fact_{Generator.ad_count}__"
        Generator.ad_count += 1
        acc_prob : float = 0
        line = line.split(';')  # type: ignore
        for index, el in enumerate(line):
            head = el.split('::')[1].replace(' ','')
            if head.endswith('.'):
                head = head[:-1]
            prob = float(el.split('::')[0])
            acc_prob += prob
            if acc_prob > 1:
                print_error_and_exit(f"Probability exceeding 1 in disjunction {line}")
            if index < len(line) - 1:
                body = f"{name}{index},"
            else:
                body = ""
            for i in range(0,len(prob_list)):
                body += f"not {name}{i},"
            body = body[:-1]

            new_clauses.append(f"{head}:- {body}.")
            cp = 1
            for p in prob_list:
                cp *= (1-p)
            if index < len(line) - 1:
                new_facts.append(f"{prob/cp}::{name}{index}.")
            prob_list.append(prob)
        return new_facts, new_clauses
    
    
    @staticmethod
    def create_intersections(intervals : 'dict[str,list[ComparisonPredicate]]') -> 'list[list[float]]':
        '''
        Creates the intervals for every continuous variable by
        joining the ranges.
        '''
        all_bounds : 'list[list[float]]' = []
        for el in intervals:
            bounds : 'list[float]' = []
            for comparison_pred in intervals[el]:
                bounds.append(comparison_pred.bound1)
                if comparison_pred.bound2 != -math.inf:
                    bounds.append(comparison_pred.bound2)
            all_bounds.append(copy.deepcopy(sorted(list(set(bounds)))))
        return all_bounds


    @staticmethod
    def generate_switch_clauses(
        bounds: 'list[list[float]]',
        continuous_facts: 'dict[str, tuple[str,float,float]]'
    ) -> 'tuple[list[list[str]],list[list[str]]]':
        '''
        Generates the clauses for the ranges.
        bound[0] = [0.4,0.5,0.7]
        continuous_facts[a] = gaussian(0,1)
        result:
        __h_a_0__ :- __fa0__.
        __h_a_1__ :- not __fa0__, __fa1__.
        __h_a_2__ :- not __fa0__, not __fa1__.
        # P(X < 0.4).
        0.6554::__fa0__.
        # P(0.4 < X < 0.5) / (1 - P(X < 0.4)) = 0.036 / (1 - 0.6554) = 0.10459
        0.10446::__fa1__.
        # P(0.5 < X < 0.7) / (1 - P(X < 0.5)) = 0.066 / (1 - 0.6915) = 0.2157724
        '''

        # names = list(continuous_facts.keys())
        # print(continuous_facts)
        # print(bounds)
        prob_facts_list : 'list[list[str]]' = []
        aux_facts_clauses : 'list[list[str]]' = []
        # bounds = [-math.inf] + bounds + [math.inf]
        for i, current_fact in zip(range(0, len(bounds)), continuous_facts):
            cb = [-math.inf] + bounds[i] + [math.inf]
            # print(cb)
            vals : 'list[float]' = []   
            for ii in range(0, len(cb) - 1):
                if continuous_facts[current_fact][0] == "gaussian":
                    vals.append(
                        continuous_cdfs.evaluate_gaussian(
                            continuous_facts[current_fact][1],
                            continuous_facts[current_fact][2],
                            cb[ii],
                            cb[ii + 1]
                        )
                    )
                elif continuous_facts[current_fact][0] == "uniform":
                    vals.append(
                        continuous_cdfs.evaluate_uniform(
                            continuous_facts[current_fact][1],
                            continuous_facts[current_fact][2],
                            cb[ii],
                            cb[ii + 1]
                        )
                    )
                elif continuous_facts[current_fact][0] == "gamma":
                    vals.append(
                        continuous_cdfs.evaluate_gamma(
                            continuous_facts[current_fact][1],
                            continuous_facts[current_fact][2],
                            cb[ii], 
                            cb[ii + 1]
                        )
                    )
                elif continuous_facts[current_fact][0] == "exponential":
                    vals.append(
                        continuous_cdfs.evaluate_exponential(
                            continuous_facts[current_fact][1],
                            cb[ii],
                            cb[ii+1]
                        )
                    )

            # print(vals)
            # generate the probabilistic facts
            t_prob_facts_list : 'list[str]' = []
            t_aux_facts_clauses : 'list[str]' = []
            name_pf = f"__{current_fact}{0}__"
            v0 = "{:.10f}".format(vals[0])
            t_prob_facts_list.append(f"{v0}::{name_pf}.")
            t_aux_facts_clauses.append(f"__int0_{current_fact}__:- {name_pf}.")
            for ii in range(1, len(vals)):
                ts = ""
                tp = 0
                for iii in range(0, ii):
                    tp += vals[iii]
                    ts += f"not __{current_fact}{iii}__,"

                if tp >= 1:  # to avoid tp > 1 due to floating point issues
                    tp = tp - 10e-6
                tp = "{:.10f}".format(vals[ii] / (1-tp))
                if float(tp) > 1: # to avoid tp > 1 due to floating point issues
                    tp = 1 - 10e-6
                name_pf = f"__{current_fact}{ii}__"
                if ii != len(vals) - 1:
                    t_prob_facts_list.append(f"{tp}::{name_pf}.")
                    t_aux_facts_clauses.append(f"__int{ii}_{current_fact}__:- {ts[:-1]}, {name_pf}.")
                else:
                    # the last one is not all the previous
                    t_aux_facts_clauses.append(
                        f"__int{ii}_{current_fact}__:- {ts[:-1]}.")

            prob_facts_list.append(copy.deepcopy(t_prob_facts_list))
            aux_facts_clauses.append(copy.deepcopy(t_aux_facts_clauses))

        # print(prob_facts_list,sep='\n')
        # print(aux_facts_clauses, sep='\n')
        return prob_facts_list, aux_facts_clauses
