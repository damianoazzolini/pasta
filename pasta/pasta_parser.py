'''
Class defining a parser for a PASTA program.
'''
from io import TextIOWrapper
import os
import re
import copy
import math

import utils
from generator import Generator
from generator import ComparisonPredicate


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


    def parse(self, from_string: str = "", approximate_version : bool = False) -> None:
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

        self.parse_program(approximate_version)

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
                

    def parse_program(self, approximate_version : bool = False) -> None:
        '''
        Second layer of program parsing: generates the ASP encoding
        for the probabilistic, abducible, map, ... facts
        '''
        # n_probabilistic_facts = 0
        gen = Generator()
        for line in self.lines_original:
            if "::" in line and not line.startswith("map"):
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
            elif line.startswith("decision"):
                fact = line.split('decision')[1][:-1].strip()
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
            elif utils.is_number(line.split(':-')[0]):
                # probabilistic IC p:- body.
                # print("prob ic")
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
                # TODO: below(a,0.5), not w. became below(a,0.5),notw,
                # so the not disapears: FIX, due to the next line
                line = line.replace(' ','') # remove the spaces
                args_cp = extract_arguments_comparison_predicates(line)
                # print(args_cp)
                # replace between(a,L,U) with below(a,U) and above(a,L)
                # and outside(a,L,U) with below(a,L) and above(a,U)
                # [above, below, between, outside]
                for el in args_cp[2]:
                    # between
                    el = el.split(',')
                    lb = float(el[1])
                    ub = float(el[2])
                    if lb > ub:
                        utils.print_error_and_exit(f"Error: {lb} > {ub}")
                    converted = f"above({el[0]},{lb}), below({el[0]},{ub})"
                    line = line.replace(f"between({el[0]},{el[1]},{el[2]})",converted)
                for el in args_cp[3]:
                    # outside: it must be substituted with two clauses
                    el = el.split(',')
                    lb = float(el[1])
                    ub = float(el[2])
                    if lb > ub:
                        utils.print_error_and_exit(f"Error: {lb} > {ub}")
                    converted_above = f"above({el[0]},{ub})"
                    converted_below = f"below({el[0]},{lb})"
                    # print("--- Converted: ---")
                    # print(line)
                    # print(f"outside({el[0]},{el[1]},{el[2]})")
                    line_above = line.replace(f"outside({el[0]},{el[1]},{el[2]})", converted_above)
                    line_below = line.replace(f"outside({el[0]},{el[1]},{el[2]})", converted_below)
                    # print(line_above)
                    # print(line_below)
                    line = line_above + "\n" + line_below
                    # print(line)
                # print(args_cp)
                # print(args_cp)
                self.insert_comparison(args_cp)
                self.lines_prob.append(line)
            else:
                if not line.startswith("#show"):
                    self.lines_prob.append(line)

        if len(self.continuous_facts) > 0:
            inter = gen.create_intersections(self.intervals)
            # print(inter)
            
            prob_facts_converted, aux_facts_clauses = gen.generate_switch_clauses(
                inter, self.continuous_facts)
            for lf in prob_facts_converted:
                for f in lf:
                    probability, fact = check_consistent_prob_fact(f, self.lpmln)
                    self.add_probabilistic_fact(fact, probability)
            for aux in aux_facts_clauses:
                self.lines_prob.extend(aux)
            # print(prob_facts_converted)
            # print(prob_facts_converted,sep='\n')
            # print(aux_facts_clauses, sep='\n')


        if not self.query and len(self.decision_facts) == 0:
            utils.print_error_and_exit("Missing query")

        i = 0
        for fact in self.probabilistic_facts:
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

        if len(self.continuous_facts) > 0:
            self.convert_comparison_predicates(gen, True)
            self.convert_comparison_predicates(gen, False)

        # print('-----')
        # for l in self.lines_prob:
        #     print(l)
        # import sys
        # sys.exit()
    
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
        lines: list[str] = []

        if self.filename == "":
            lines = from_string.split('\n')
        else:
            fp = open(self.filename, "r")
            lines = fp.readlines()
            fp.close()

        i = 0
        program = ""
        # target = ""
        prob_facts_dict: dict[str, float] = dict()
        interpretations_dict: dict[int, list[str]] = dict()

        training_set: list[list[str]] = []
        test_set: list[list[str]] = []

        train_ids: list[int] = []
        test_ids: list[int] = []

        offset = 0

        while i < len(lines):
            lines[i] = lines[i].replace('\n', '')
            if lines[i].startswith("#program('"):
                i = i + 1
                while(not (lines[i].startswith("')."))):
                    program = program + lines[i]
                    # look for prob facts in the program that need to be considered
                    # in the dict but whose probabilities cannot be set
                    if '::' in lines[i]:
                        prob_fact = lines[i].split('::')[1].replace(
                            '\n', '').replace('.', '').replace(' ', '')
                        prob_facts_dict[prob_fact] = float(lines[i].split('::')[0])
                        offset = offset + 1
                    i = i + 1
            elif lines[i].startswith("#learnable("):
                ll = lines[i].split("#learnable(")
                name = ll[1].replace('\n', '')[:-2]
                prob_facts_dict[name] = 0.5
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

        return training_set, test_set, program, prob_facts_dict, offset


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
        Returns a string that represent the ASP program obtained by converting the PASP
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
            returns a string that represent the ASP program where models 
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
        if key in self.probabilistic_facts:
            utils.error_prob_fact_twice(key, prob, self.probabilistic_facts[key])
        self.probabilistic_facts[key] = float(prob)


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
