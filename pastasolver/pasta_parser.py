'''
Class defining a parser for a PASTA program.
'''
from io import TextIOWrapper
import os
import re
import copy
import math
import sys
from scipy.optimize import minimize, fsolve
import scipy.stats
from scipy import special

from . import utils
from .generator import Generator
from .generator import ComparisonPredicate
# from .generator import ParametersFinder
from .continuous_cdfs import *

def symbol_endline_or_space(char1: str) -> bool:
    return char1 == '\n' or char1 == '\r' or char1 == '\r\n' or char1 == '\n\r' or char1 == ' '


def endline_symbol(char1: str) -> bool:
    return char1 == '\n' or char1 == '\r\n' or char1 == '\n\r'


def check_consistent_prob_fact(line_in: str, lpmln: bool = False) -> 'tuple[float, str]':
    if lpmln:
        r = r"[0-9]+::[a-z_][a-z_0-9]*(\([a-z_0-9]*(,[a-z_0-9]*)*\))*\."
    else:
        r = r"0\.[0-9]+::[a-z_][a-z_0-9]*(\([a-z_0-9]*(,[a-z_0-9]*)*\))*\."
    
    # TODO: this is marked as incorrect
    # Error: Probabilistic fact ->0.3::shops(a(0)).<- ill formed
    x = re.match(r, line_in.strip())
    if x is None:
        utils.print_error_and_exit(
            f"Probabilistic fact ->{line_in}<- ill formed")

    line = line_in.split("::")

    return float(line[0]), line[1][:-1]


def get_functor(term: str) -> 'tuple[str,int]':
    '''
    Extracts the functor from a compound term.
    '''
    # clean up choice rules m{f}n
    t1 = term.split('{')
    term = t1[len(t1) - 1]
    t1 = term.split('}')
    term = t1[0]
    
    return term.split('(')[0], term.count(',') + 1 if '(' in term else 0


def get_fact_and_utility(term: str) -> 'tuple[str,float]':
    '''
    Extracts the utility and the term from utility(term,utility).
    '''
    t = term.split("utility")[1][1:-2] # eat ). and the initial (
    i = len(t) - 1
    while t[i] != ',' and i > 0:
        i = i - 1
    return t[0:i], float(t[i+1:])


def extract_arguments_comparison_predicates(line : str) -> 'list[list[str]]':
    '''
    From a line extracts all the arguments of the comparison predicates.
    Es: line = "above(a,4), r, above(gs, 9), below(g,7), between(a,1,4)."
    Return: [['a,4', 'gs, 9'], ['g,7'], ['a,1,4'], []]
    Order = ["above", "below", "between", "outside"]
    '''
    comparison = ["above", "below", "between", "outside"]
    preds_bound : 'list[list[str]]'= []
    for cp in comparison:
        indexes = [m.start() for m in re.finditer(cp, line)]
        tl : 'list[str]' = []
        for i in indexes:
            pos = i + len(cp) + 1
            init = pos
            end = -1
            pars = 1
            while pars > 0 and pos < len(line):
                if line[pos] == ')':
                    pars -= 1
                elif line[pos] == '(':
                    pars += 1
                pos += 1
            end = pos - 1
            tl.append(line[init:end])
        preds_bound.append(copy.deepcopy(tl))

    return preds_bound


class PastaParser:
    '''
    Parameters:
        - filename: name of the file to read
        - query: query
        - evidence: evidence
        - lines_original: lines from the parsing of the original file
        - lines_prob: lines obtained by parsing probabilistic facts
        - probabilistic_fact: dictionary containing pairs [probabilistic fact, probability]
        - abducibles: list of abducibles
    '''

    def __init__(
        self,
        filename : str,
        query : str = "",
        evidence : str = "",
        for_asp_solver : bool = False,
        naive_dt : bool = False,
        lpmln : bool = False
        ) -> None:
        self.filename : str = filename
        self.query : str = query
        self.evidence : str = evidence
        self.lines_original : 'list[str]' = []
        self.lines_prob : 'list[str]' = []
        self.probabilistic_facts : 'dict[str,float]' = {} # pairs [fact,prob]
        self.abducibles : 'list[str]' = []
        self.n_probabilistic_ics : int = 0
        self.body_probabilistic_ics : 'list[str]' = []
        self.map_id_list : 'list[int]' = []
        self.fact_utility : 'dict[str,float]' = {}
        self.decision_facts : 'list[str]' = []
        self.continuous_facts : 'dict[str,tuple[str,float,float]]' = {}
        self.intervals : 'dict[str,list[ComparisonPredicate]]' = {}
        self.lpmln : bool = lpmln
        self.for_asp_solver : bool = for_asp_solver
        self.naive_dt : bool = naive_dt
        self.optimizable_facts : 'dict[str,tuple[float,float]]' = {}
        self.reducible_facts : 'dict[str,float]' = {}
        self.objective_function : str = ""
        self.constraints_list : 'list[str]' = []


    def insert_comparison(self, args_list : 'list[list[str]]') -> None:
        '''
        Insert a comparison predicate in the list.
        '''
        names = ["above","below","between","outside"]
        for i, name in zip(range(0, len(args_list)), names):
            for el in args_list[i]:
                el = el.split(',')
                cp = ComparisonPredicate(name, float(el[1]), float(el[2]) if len(el) == 3 else -math.inf)
                if el[0] in self.intervals:
                    self.intervals[el[0]].append(cp)
                else:
                    self.intervals[el[0]] = [cp]


    def convert_comparison_predicates(self, gen : Generator, below : bool) -> None:
        '''
        Explodes the comparison predicates and generates the
        new clauses.
        '''
        for el, interval in zip(self.intervals, gen.create_intersections(self.intervals)):
            to_remove : 'list[int]' = [] # lines to remove (with comparison predicates)
            new_lines : 'list[str]' = [] # lines to insert
            if below is False:
                interval.reverse() # above: reversed
            # print(interval)
            for interval_index in range(0, len(interval)):
                cp = "below" if below else "above"
                current_comparison_predicate = f'{cp}({el},{interval[interval_index]})'
                # print(current_comparison_predicate)
                for line_index in range(0, len(self.lines_prob)):
                    if current_comparison_predicate in self.lines_prob[line_index]:
                        to_remove.append(line_index)
                        for sub_interval in range(0, interval_index + 1):
                            pos = sub_interval if below else (len(interval) - sub_interval)
                            new_lines.append(self.lines_prob[line_index].replace(
                                current_comparison_predicate, f"__int{pos}_{el}__"))

            # remove the old lines and insert the new ones
            for index in sorted(to_remove, reverse=True):
                del self.lines_prob[index]
            self.lines_prob.extend(new_lines)

    
    def get_file_handler(self, from_string : str = "") -> TextIOWrapper:
        if not from_string:
            if not os.path.isfile(self.filename):
                utils.print_error_and_exit(f"File {self.filename} not found")
            return open(self.filename, "r")
        else:
            import io
            return io.StringIO(from_string)


    def parse(self,
              from_string: str = "",
              approximate_version : bool = False,
              keep_hybrid : bool = False) -> None:
        '''
        Parses the file
        '''
        l2 : 'list[str]' = []
        heads : 'list[str]' = []
        
        f = self.get_file_handler(from_string)
        lines = f.readlines()
        f.close()
        # hack since a line not terminated with \n is not considered
        # so ignored by the next loop. IMPROVE.
        lines[-1] = lines[-1] + '\n' 

        # https://stackoverflow.com/questions/68652859/how-to-exclude-floating-numbers-from-pythonss-regular-expressions-that-splits-o
        for l in lines:
            if not l.lstrip().startswith('%'):
                ll = re.findall(r"\S.*?(?:[?!\n]|(?<!\d)\.(?!\d))", l)
                l2.extend(ll)

        i = 0
        while i < len(l2):
            line = l2[i].replace('\n','').replace('\r','')
            
            l1 : str = ""
  
            if not line.rstrip().endswith('.'):
                # to consider clauses that spans multiple lines
                while not line.rstrip().endswith('.') and i < len(l2):
                    percent = line.find('%')
                    if percent != -1:
                        line = line[:percent]
                    l1 += line
                    i = i + 1
                    line = l2[i].replace('\n', '').replace('\r', '')
                percent = line.find('%')
                if percent != -1:
                    line = line[:percent]
                l1 += line
                i = i + 1
            else:
                l1 = line
                i = i + 1

            self.lines_original.append(l1)

        self.parse_program(approximate_version, keep_hybrid)

        for el in self.lines_prob:
            if ':-' in el:
                h = el.split(':-')[0]
                if len(h) > 0: # filter out constraints
                    for hh in h.split(';'):
                        heads.append(hh.replace(' ',''))

        # check for clauses with a prob fact in the head
        for pf in self.probabilistic_facts.keys():
            for h in heads:
                if get_functor(h) == get_functor(pf):
                    utils.print_error_and_exit(f"Cannot use the probabilistic fact {pf} as head of a rule.")

        # check for clauses with q or nq or 3 or ne in the head
        for h in heads:
            if h in ("q", "nq", "e", "ne"):
                utils.print_error_and_exit(
                    f"Cannot use {h} as head of a rule.")
                

    def parse_program(self,
                      approximate_version : bool = False,
                      keep_hybrid : bool = False) -> None:
        '''
        Second layer of program parsing: generates the ASP encoding
        for the probabilistic, abducible, map, ... facts
        '''
        # n_probabilistic_facts = 0
        gen = Generator()
        for line in self.lines_original:
            if "::" in line and not line.startswith("map") and not line.startswith("optimizable") and not line.startswith("reducible") and "?::" not in line:
                if ':-' in line:
                    utils.print_error_and_exit("Probabilistic clauses are not supported\n" + line)
                if ';' in line:
                    new_facts, new_clauses = Generator.generate_facts_from_disjunction(line)
                    self.lines_prob.extend(new_clauses)
                    for f in new_facts:
                        probability, fact = check_consistent_prob_fact(f, self.lpmln)
                        self.add_probabilistic_fact(fact, probability)
                        # n_probabilistic_facts = n_probabilistic_facts + 1
                else:
                    probability, fact = check_consistent_prob_fact(line.replace(' ',''), self.lpmln)
                    self.add_probabilistic_fact(fact,probability)
                    # n_probabilistic_facts = n_probabilistic_facts + 1
            elif ':' in line and ((":gaussian(" in line) or (":exponential(" in line) or (":uniform(" in line) or (":gamma(" in line)):
                # continuous fact with 2 arguments
                if ("gaussian(" in line) or ("uniform(" in line) or (":gamma(" in line):
                    distr_type = ""
                    if "gaussian(" in line:
                        line = line.split(":gaussian(")
                        distr_type = "gaussian"
                    elif "uniform(" in line:
                        line = line.split(":uniform(")
                        distr_type = "uniform"
                    elif "gamma(" in line:
                        line = line.split(":gamma(")
                        distr_type = "gamma"

                    name = line[0]
                    parameters = line[1][:-2].split(',') # remove ).
                    if len(parameters) != 2:
                        utils.print_error_and_exit(f"The distribution of {name} requires two parameters.")
                    try:
                        p0 = float(parameters[0])
                        p1 = float(parameters[1])
                    except:
                        utils.print_error_and_exit(f"Error in parameters {parameters}.")
                    # i need to store also the type of distribution
                    self.continuous_facts[name] = (distr_type, p0, p1)
                elif "exponential(" in line:
                    line = line.split(":exponential(")
                    name = line[0]
                    parameters = line[1][:-2].split(',')  # remove ).
                    if len(parameters) != 1:
                        utils.print_error_and_exit(
                            "Exponential distribution requires one parameter.")
                    try:
                        rate = float(parameters[0])
                    except:
                        utils.print_error_and_exit(
                            f"Error in parameters {parameters}.")
                    self.continuous_facts[name] = ("exponential", rate, -1)
            elif line.startswith("query("):
                # remove the "query" functor and handles whether the line
                # does not terminate with .
                # query(fly(1)) -> fly(1)
                if line[-1] == ".":
                    self.query = line.split("query")[1][:-2][1:]
                else:
                    self.query = line.split("query")[1][:-1][1:]
            elif line.startswith("evidence("):
                if line[-1] == ".":
                    # remove the "evidence" functor and handles whether the line
                    # does not terminate with .
                    # evidence(fly(1)) -> fly(1)
                    self.evidence = line.split("evidence")[1][:-2][1:]
                else:
                    self.evidence = line.split("evidence")[1][:-1][1:]
            elif line.startswith("("):
                expanded_conditional = gen.generate_clauses_for_conditionals(line)
                self.lines_prob.extend(expanded_conditional)
            elif line.startswith("abducible"):
                _, abducible = gen.generate_clauses_for_abducibles(line, 0)
                # self.lines_prob.append(clauses)
                # self.abducibles.append(abducible)
                self.abducibles.append(abducible)
            elif line.startswith("map"):
                # add the MAP fact as probabilistic
                fact = line.split('map')[1]
                probability, fact = check_consistent_prob_fact(fact)
                self.map_id_list.append(len(self.probabilistic_facts))
                self.add_probabilistic_fact(fact,probability)
            elif line.startswith("decision") or line.startswith("?::"):
                if line.startswith("decision"):
                    to_s = "decision"
                else:
                    to_s = "?::"
                fact = line.split(to_s)[1][:-1].strip()
                clauses = gen.generate_clauses_for_dt(fact, "decision", self.naive_dt)
                self.decision_facts.append(fact)
                self.lines_prob.extend(clauses)
            elif line.startswith("utility"):
                fact, utility = get_fact_and_utility(line)
                self.fact_utility[fact] = utility
                # print(f"utility({fact},{int(utility)}):- {fact}.")
                # keep it to possibly impose ASP constraints
                # on the utilites (e.g. on weights?)
                self.lines_prob.append(line)
                clauses = gen.generate_clauses_for_dt(fact, "utility", self.naive_dt)
                # self.decision_facts.append(fact)
                self.lines_prob.extend(clauses)
            elif line.startswith("optimizable"):
                fact_and_range = line.split('optimizable')[1].replace(' ','')
                prob_range = fact_and_range.split('::')[0].replace('[','').replace(']','')
                fact = fact_and_range.split('::')[1].replace('.','')
                lower_bound_prob = float(prob_range.split(',')[0])
                upper_bound_prob = float(prob_range.split(',')[1])
                k = f"P({fact.replace('(','_').replace(')','_').replace(',','_')})"
                # self.optimizable_facts[fact.replace('(','_').replace(')','_').replace(',','_')] = (lower_bound_prob,upper_bound_prob)
                self.optimizable_facts[k] = (lower_bound_prob,upper_bound_prob)
                self.add_probabilistic_fact(fact, 0.5)
                # to_add : 'list[str]' = []
            elif line.startswith("reducible"):
                fact = line.split('reducible')[1]
                if '::' in fact:
                    probability, fact = check_consistent_prob_fact(fact)
                else:
                    fact = fact[:-1].replace(' ','')
                    probability = 1
                self.reducible_facts[fact.replace('(','_').replace(')','_').replace(',','_')] = probability
                self.add_probabilistic_fact(fact, probability)
            elif utils.is_number(line.split(':-')[0]):
                # probabilistic IC p:- body.
                # generate the probabilistic fact
                new_line = line.split(':-')[0] + "::icf" + str(self.n_probabilistic_ics) + "."
                probability, fact = check_consistent_prob_fact(new_line)
                self.add_probabilistic_fact(fact, probability)
                new_clause = "ic" + str(self.n_probabilistic_ics) + ":- " + line.split(':-')[1]
                self.lines_prob.append(new_clause)

                new_ic_0 = ":- icf" + str(self.n_probabilistic_ics) + ", ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append(new_ic_0)

                new_ic_1 = ":- not icf" + str(self.n_probabilistic_ics) + ", not ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append(new_ic_1)

                self.n_probabilistic_ics = self.n_probabilistic_ics + 1
            elif ("above(" in line) or ("below(" in line) or ("between(" in line) or ("outside(" in line):
                # comparison predicates
                # for simplicity, suppose that the variables are atoms and
                # not compound
                # hack to preserve the not before space: terrible
                line = line.replace('not ','**not**')
                line = line.replace(' ','') # remove the spaces
                line = line.replace('**not**','not ')
                args_cp = extract_arguments_comparison_predicates(line)
                # replace between(a,L,U) with below(a,U) and above(a,L)
                # and outside(a,L,U) with below(a,L) and above(a,U)
                # [above, below, between, outside]
                for el in args_cp[2]:
                    # between
                    el = el.split(',')
                    lb = float(el[1])
                    ub = float(el[2])
                    if lb > ub:
                        utils.print_error_and_exit(f"{lb} > {ub}")
                    converted = f"above({el[0]},{lb}), below({el[0]},{ub})"
                    line = line.replace(f"between({el[0]},{el[1]},{el[2]})",converted)
                for el in args_cp[3]:
                    # outside: it must be substituted with two clauses
                    el = el.split(',')
                    lb = float(el[1])
                    ub = float(el[2])
                    if lb > ub:
                        utils.print_error_and_exit(f"{lb} > {ub}")
                    converted_above = f"above({el[0]},{ub})"
                    converted_below = f"below({el[0]},{lb})"
                    line_above = line.replace(f"outside({el[0]},{el[1]},{el[2]})", converted_above)
                    line_below = line.replace(f"outside({el[0]},{el[1]},{el[2]})", converted_below)
                    line = line_above + "\n" + line_below
                self.insert_comparison(args_cp)
                self.lines_prob.append(line)
            else:
                if not line.startswith("#show"):
                    self.lines_prob.append(line)

        if len(self.continuous_facts) > 0 and not keep_hybrid:
            inter = gen.create_intersections(self.intervals)
            prob_facts_converted, aux_facts_clauses = gen.generate_switch_clauses(
                inter, self.continuous_facts)
            for lf in prob_facts_converted:
                # print("Probabilistic facts converted")
                # print(lf)
                for f in lf:
                    probability, fact = check_consistent_prob_fact(f, self.lpmln)
                    self.add_probabilistic_fact(fact, probability)
            for aux in aux_facts_clauses:
                self.lines_prob.extend(aux)

        if not self.query and len(self.decision_facts) == 0:
            utils.print_error_and_exit("Missing query")

        external_names : 'list[str]' = []
        original_names : 'list[str]' = []
        if keep_hybrid:
            for fact in self.intervals:
                # create external names
                for el in self.intervals[fact]:
                    if el.comparison_type == "above" or el.comparison_type == "below":
                        v = f"{fact}_{el.comparison_type}_{str(el.bound1).replace('.','_')}"
                        o = f"{el.comparison_type}({fact},{el.bound1})"
                        external_names.append(v)
                        original_names.append(o)
                        self.add_probabilistic_fact(v, 0.5)
                    elif el.comparison_type == "between" or el.comparison_type == "outside":
                        # this since between and outside are converted into above and below in a 
                        # previous step
                        ct0 = "above" if el.comparison_type == "between" else "below"
                        ct1 = "below" if el.comparison_type == "between" else "above"

                        v0 = f"{fact}_{ct0}_{str(el.bound1).replace('.','_')}"
                        o0 = f"{ct0}({fact},{el.bound1})"
                        v1 = f"{fact}_{ct1}_{str(el.bound2).replace('.','_')}"
                        o1 = f"{ct1}({fact},{el.bound2})"

                        external_names.append(v0)
                        external_names.append(v1)
                        original_names.append(o0)
                        original_names.append(o1)
                        self.add_probabilistic_fact(v0, 0.5)
                        self.add_probabilistic_fact(v1, 0.5)

            # replace each occurrence of comparison predicates with the external facts
            for original, external in zip(original_names,external_names):
                for i in range(0,len(self.lines_prob)):
                    if original in self.lines_prob[i]:
                        self.lines_prob[i] = self.lines_prob[i].replace(original, external)

        i = 0
        facts = list(self.probabilistic_facts.keys()) + external_names
        for fact in facts:
            if self.for_asp_solver and i in self.map_id_list:
                clauses = gen.generate_clauses_for_facts_for_asp_solver(
                    i, fact, self.probabilistic_facts[fact])
            else:
                clauses = gen.generate_clauses_for_facts(fact, approximate_version, self.lpmln)

            for c in clauses:
                self.lines_prob.append(c)
            i = i + 1

        i = 0
        for abd in self.abducibles:
            # kind of hack, refactor generate_clauses_for abducibles TODO
            clauses, _ = gen.generate_clauses_for_abducibles("abducible " + abd + ".", i)
            i = i + 1
            self.lines_prob.extend(clauses)

        if len(self.continuous_facts) > 0 and not keep_hybrid:
            self.convert_comparison_predicates(gen, True)
            self.convert_comparison_predicates(gen, False)

    
    def inference_to_mpe(self, from_string: str = "") -> 'tuple[str,int]':
        '''
        Adds 'map' before probabilistic facts.
        '''
        f = self.get_file_handler(from_string)
        parsed_program : str = ""
        n_vars = 0
        for line in f:
            if "::" in line and not line.strip().startswith('%'):
                line = f"map {line.strip()}"
                n_vars += 1
            parsed_program = parsed_program + line + "\n"
        return parsed_program, n_vars 


    def parse_input_learning(self, from_string: str = "") -> 'tuple[list[list[str]],list[list[str]],str,dict[str,float],int]':
        '''
        #example(pos,Id,'atom') where Id is the Id of the (partial) answer set and atom is the correspondent atom
        #test(IdList)
        #train(IdList)
        #program('program') where program is a set of clauses
        #learnable(atom) where atom is a probabilistic fact with init probability 0.5
        '''
        lines: 'list[str]' = []

        if self.filename == "":
            lines = from_string.split('\n')
        else:
            fp = open(self.filename, "r")
            lines = fp.readlines()
            fp.close()

        i = 0
        program = ""
        # target = ""
        prob_facts_dict: 'dict[str, float]' = dict()
        interpretations_dict: 'dict[int, list[str]]' = dict()

        training_set: 'list[list[str]]' = []
        test_set: 'list[list[str]]' = []

        train_ids: 'list[int]' = []
        test_ids: 'list[int]' = []
        
        # offset = 0

        while i < len(lines):
            lines[i] = lines[i].replace('\n', '')
            if lines[i].startswith("#program('"):
                i = i + 1
                while(not (lines[i].startswith("')."))):
                    program = program + lines[i]
                    i = i + 1
            elif lines[i].startswith("#learnable("):
                # convert the continuous probabilistic facts
                #learnable(a,gaussian). -> a:gaussian(0,1).
                to_split = ""
                # this if-else is terrible
                if "gaussian" in lines[i]:
                    to_split = "gaussian"
                elif "uniform" in lines[i]:
                    to_split = "uniform"
                elif "gamma" in lines[i]:
                    to_split = "gamma"
                elif "exponential" in lines[i]:
                    to_split = "exponential"
                else:
                    # discrete probabilistic fact
                    ll = lines[i].split("#learnable(")
                    name = ll[1].replace('\n', '')[:-2]
                    prob_facts_dict[name] = 0.5
                
                if to_split:
                    # print(lines[i])
                    line = lines[i].replace('\n', '')[:-2].split("#learnable(")[1]
                    # print(line)
                    distr_args = ""
                    if to_split == "exponential":
                        distr_args = "(1)"
                    else:
                        distr_args = "(0,1)"
                    pf = f"{line.split(to_split)[0][:-1]}:{to_split}{distr_args}"
                    # ora devo convertire
                    prob_facts_dict[pf] = -1
                    program += pf + '.\n'
                    # sys.exit()
                
                i = i + 1
            elif lines[i].startswith("#positive("):
                ll = lines[i].split("#positive(")
                id_interpretation = int(ll[1].split(',')[0])
                atom = ll[1].replace('\n', '')[len(str(id_interpretation)) + 1: -2]
                if id_interpretation in interpretations_dict.keys():
                    interpretations_dict[id_interpretation].append(atom)
                else:
                    interpretations_dict[id_interpretation] = [atom]
                i = i + 1
            elif lines[i].startswith("#negative("):
                ll = lines[i].split("#negative(")
                id_interpretation = int(ll[1].split(',')[0])
                atom = ll[1].replace('\n', '')[len(str(id_interpretation)) + 1: -2]
                if id_interpretation in interpretations_dict.keys():
                    interpretations_dict[id_interpretation].append(f"not {atom}")
                else:
                    interpretations_dict[id_interpretation] = [f"not {atom}"]

                i = i + 1
            elif lines[i].startswith("#train("):
                ll = lines[i].split("#train(")
                train_ids = list(map(int, ll[1].replace('\n', '')[:-2].split(',')))
                i = i + 1
            elif lines[i].startswith("#test("):
                ll = lines[i].split("#test(")
                test_ids = list(map(int, ll[1].replace('\n', '')[:-2].split(',')))
                i = i + 1
            else:
                i = i + 1

        for id in train_ids:
            training_set.append(interpretations_dict[int(id)])

        for id in test_ids:
            test_set.append(interpretations_dict[int(id)])
        
        # recycle the parsing from the parse method
        self.parse(from_string=program)
        
        # self.probabilistic_facts contains probabilistic facts already in the program
        # prob_facts_dict contains the probabilistic facts whose probabilities should be learned
        
        # merge the two dicts: the one for the prob facts in the
        # program and the one for the prob facts to learn
        df = dict(self.probabilistic_facts, **prob_facts_dict)
        
        # filter out the #show statements that parse adds
        prg = [l for l in self.lines_prob if not l.startswith('#')]
        # add back the probabilistic facts
        for el in self.probabilistic_facts:
            prg.append(f'{self.probabilistic_facts[el]}::{el}.')
        # remove extra statements added by parse
        for el in self.probabilistic_facts:
            prg.remove('0{' + el + '}1.')
            prg.remove(f"not_{el}:- not {el}.")

        # i need to move from self.probabilistic_facts to prob_facts_dict the
        # probabilistic facts describing continuous random variables
        for el in self.probabilistic_facts:
            if el.startswith('__'):
                # check whether it is a fact to be learned
                to_learn = False
                for el_to_learn in prob_facts_dict:
                    if ':' in el_to_learn and el_to_learn.split(':')[0] in el:
                        to_learn = True
                        break
                if to_learn:
                    # move
                    prob_facts_dict[el] = self.probabilistic_facts[el]
        
        # get the dict with pf to learn
        intersect = list(set(self.probabilistic_facts.keys()) & set(prob_facts_dict.keys()))
        for pf in intersect:
            del self.probabilistic_facts[pf]

        # remove from df the continuous facts
        for pf in prob_facts_dict:
            if ':' in pf:
                del df[pf]

        # return training_set, test_set, program, prob_facts_dict, offset
        return training_set, test_set, '\n'.join(prg) + '\n', df, len(self.probabilistic_facts)


    def get_content_to_compute_minimal_set_facts(self) -> 'list[str]':
        '''
        Parameters:
            - None
        Returns:
            - str: program used to compute the minimal set of probabilistic
            facts to make the query true
        Behavior:
            generate the file to pass to ASP to compute the minimal set
            of probabilistic facts to make the query true
        '''
        return self.lines_prob + [":- not " + self.query + "."] if self.evidence == "" else self.lines_prob + [":- not " + self.evidence + "."]


    def get_asp_program(self, lpmln : bool = False) -> 'list[str]':
        '''
        Returns a string that represent the answer set program obtained by converting the PASP
        '''
        if self.query and not lpmln:
            self.lines_prob.extend([f"q:- {self.query}.","#show q/0.",f"nq:- not {self.query}.","#show nq/0."])

            if self.evidence:
                self.lines_prob.extend([f"e:- {self.evidence}.","#show e/0.",f"ne:- not {self.evidence}.","#show ne/0."])

        return list(set(self.lines_prob))


    def get_asp_program_approx(self) -> 'list[str]':
        '''
        Parameters:
            - None
        Returns:
            - str: string representing the program that can be used to
            compute lower and upper probability
        Behavior:
            returns a string that represent the answer set program where models
            need to be computed
        '''
        if self.evidence == "":
            self.lines_prob.append(f"q:- {self.query}.")
            self.lines_prob.append("#show q/0.")
            self.lines_prob.append(f"nq:- not {self.query}.")
            self.lines_prob.append("#show nq/0.")
        else:
            self.lines_prob.append(f"qe:- {self.query}, {self.evidence}.")
            self.lines_prob.append("#show qe/0.")
            self.lines_prob.append(f"nqe:- not {self.query}, {self.evidence}.")
            self.lines_prob.append("#show nqe/0.")

        return list(set(self.lines_prob))


    def add_probabilistic_fact(self, term : str, prob : float) -> None:
        '''
        Adds the current probabilistic fact and its probability in the 
        list of probabilistic facts
        '''
        key = term.split('.')[0]
        if key in self.probabilistic_facts and self.probabilistic_facts[key] != prob:
            utils.error_prob_fact_twice(key, prob, self.probabilistic_facts[key])
        self.probabilistic_facts[key] = float(prob)


    def reconstruct_parameters(self, learned_prob_facts_dict : 'dict[str,float]'):
        '''
        Reconstructs the parameters of continuous distributions
        starting from a set of probabilistic facts obtained via
        parameter learning.
        '''
        
        def evaluate_fn_gaussian(mv, *args):
            '''
            Target of the minimization process for gaussian distributions.
            '''
            # mv[0] = mean
            # mv[1] = variance

            to_minimize : float = 0
            for el in args[0]:
                x = el[0]
                y = el[1]
                # x - \mu / \sigma
                cdf = (0.5*(1 + special.erf((x - mv[0]) / (math.sqrt(2)*mv[1]))))
                to_minimize += abs(cdf - y)

            return to_minimize

        def equations_gauss(p, *args):
            '''
            Used when there are 2 values and 2 parameters.
            '''
            x0, y0 = p
            funz = []
            for (p0,v0) in args[0]:
                funz.append(0.5*(1 + special.erf((p0 - x0) / (math.sqrt(2)*y0) )) - v0)
            
            return funz
        
        def evaluate_fn_gamma(mv, *args):
            '''
            Target of the minimization process for gamma distributions.
            CDF(x; \alpha, \beta): \gamma(\alpha, \beta * x) / \Gamma(\alpha)
            '''
            # mv[0] = alpha
            # mv[1] = beta

            to_minimize: float = 0
            # print(f"args: {args[0]}")
            for el in args[0]:
                # print(f"mv: {mv}")
                x = el[0]
                y = el[1]
                a, b = mv[0], mv[1]
                # x - \mu / \sigma
                cdf = scipy.stats.gamma.cdf(x, a, scale=b)
                # cdf = special.gammainc(a,b*x) / special.gamma(a)
                # print(f"cdf: {cdf}")
                to_minimize += abs(cdf - y)

            return to_minimize
        
        ### BODY
        
        computed_vals : 'dict[str,list[float]]' = {}
        
        for lpf in learned_prob_facts_dict:
            current_fact = lpf[2:-3]
            if lpf.startswith('__') and current_fact in self.intervals:
                if current_fact in computed_vals:
                    computed_vals[current_fact].append(learned_prob_facts_dict[lpf])
                else:
                    computed_vals[current_fact] = [learned_prob_facts_dict[lpf]]

        # print(computed_vals)
        for cv in computed_vals:
            bounds = self.intervals[cv]
            vals = computed_vals[cv]
            distr = self.continuous_facts[cv][0]

            realprobs = [vals[0]]
            for i in range(1, len(vals)):
                # compute the denominator
                real_val = vals[i] * (1 - sum(realprobs)) + sum(realprobs)
                realprobs.append(real_val)
            
            args_to_pass : 'list[tuple[float,float]]' = []
            for (p_i,vi) in zip(bounds,realprobs):
                args_to_pass.append((p_i.bound1,vi))
            mean_d = None
            var_d = None
            if distr == "gaussian":
                if len(args_to_pass) == 2:
                    mean_d, var_d =  fsolve(equations_gauss, x0=(0.5, 0.5), args=args_to_pass)
                else:
                    res = minimize(
                        evaluate_fn_gaussian,
                        [0.5,0.5],
                        args=args_to_pass,
                        method='Powell',
                    )
                    mean_d = res.x[0]
                    var_d = res.x[1]
                print(f"{cv}:\n\tMean: {mean_d}\n\tVariance: {var_d}")
            elif distr == "gamma":
                res = minimize(
                        evaluate_fn_gamma,
                        [0.5,0.5],
                        args=args_to_pass,
                        method='Powell',
                    )
                mean_d = res.x[0]
                var_d = res.x[1]
                print(f"{cv}:\n\tShape: {mean_d}\n\tRate: {var_d}")
                


    def __repr__(self) -> str:
        '''
        String representation of the current class
        '''
        return "filename: " + self.filename + "\n" + \
        "query: " + str(self.query) + "\n" + \
        (("evidence: " + str(self.evidence) + "\n") if self.evidence else "") + \
        "probabilistic facts:\n" + str([str(x) + " " + str(y) for x, y in self.probabilistic_facts.items()]) + "\n" + \
        "n probabilistic facts:\n" + str(self.probabilistic_facts) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "probabilities file:\n" + str(self.lines_prob) + "\n" + \
        (("abducibles: " + str(self.abducibles)) if len(self.abducibles) > 0 else "")
