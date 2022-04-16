'''
Class defining a parser for a PASTA program.
'''
import os
import sys
from typing import Union

import utils

import time
from timeit import default_timer as timer

# from matplotlib import lines
# from typing import Dict, Union

import generator

class PastaParser:
    '''
    Parameters:
        - filename: name of the file to read
        - precision: multiplier for the log probabilities
        - query: query
        - evidence: evidence
        - lines_original: lines from the parsing of the original file
        - lines_prob: lines obtained by parsing probabilistic facts
        - probabilistic_fact: dictionary containing pairs [probabilistic fact, probability]
        - abducibles: list of abducibles
    '''

    def __init__(self, filename: str, precision: int, query = None, evidence = None, identify_useless_facts = False) -> None:
        self.filename : str = filename
        self.precision : int = precision
        self.query : str = query
        self.evidence : str = evidence
        self.lines_original : list = []
        self.lines_prob : list = []
        self.probabilistic_facts = dict() # pairs [fact,prob]
        self.abducibles : list = []
        self.n_probabilistic_ics : int = 0
        self.body_probabilistic_ics : list = []

    @staticmethod
    def symbol_endline_or_space(char1: str) -> bool:
        return char1 == '\n' or char1 == '\r\n' or char1 == ' '

    @staticmethod
    def endline_symbol(char1: str) -> bool:
        return char1 == '\n' or char1 == '\r\n'

    @staticmethod
    def is_number(n: Union[int, float]) -> bool:
        try:
            float(n)
        except ValueError:
            return False
        return True

    @staticmethod
    def is_int(n: int) -> bool:
        try:
            int(n)
        except ValueError:
            return False
        return True

    # extracts the functor from a compound term. If the term is an atom
    # returns the atom itself
    @staticmethod
    def get_functor(term: str) -> str:
        r = ""
        i = 0
        while i < len(term) and term[i] != '(':
            r = r + term[i]
            i = i + 1
        return r

    @staticmethod
    def check_consistent_prob_fact(line: str) -> Union[float, str]:
        if not line.endswith('.'):
            sys.exit("Missing final . in " + line)

        line = line.split("::")
        # for example: line = ['0.5', 'f(1..3).']
        if len(line) != 2:
            sys.exit("Error in parsing: " + str(line))

        if not PastaParser.is_number(line[0]):
            print("---- ")
            sys.exit("Error: expected a float, found " + str(line[0]))

        prob = float(line[0])

        if prob > 1 or prob <= 0:
            sys.exit("Probabilities must be in the range ]0,1], found " + str(prob))

        # [:-1] to remove final .
        term = line[1][:-1]

        if len(term) == 0 or not term[0].islower():
            sys.exit("Invalid probabilistic fact " + str(term))

        return prob, term

    '''
    Parses a program into an alternative form: probabilistic 
        facts are converted into external facts
    '''
    def parse_approx(self, from_string : str = None) -> None:
        if from_string is None and os.path.isfile(self.filename) == False:
            print("File " + self.filename + " not found")
            sys.exit()

        if from_string is None:
            f = open(self.filename,"r")
        else:
            import io
            f = io.StringIO(from_string)

        lines = f.readlines()

        for l in lines:
            if not l.startswith('%'):
                if '::' in l:
                    # probabilistic fact
                    l = l.replace('\n','').replace('\t','').split('::')
                    prob = l[0]
                    term = l[1].replace('\n','').replace('\r','').replace('.','')
                    self.probabilistic_facts[term] = float(prob)
                    self.lines_prob.append(f'#external {term}.')
                elif not l.startswith('\n'):
                    self.lines_prob.append(l.replace('\n','').replace('\r',''))

        return True

    '''
    Parameters:
        - None
    Returns:
        - list of strings representing the program
    Behavior:
        - parses the file and extract the lines
    '''
    def parse(self, from_string : str = None) -> None:
        if from_string is None and os.path.isfile(self.filename) == False:
            print("File " + self.filename + " not found")
            sys.exit()
        
        if from_string is None:
            f = open(self.filename,"r")
        else:
            import io
            f = io.StringIO(from_string)
        
        char = f.read(1)
        if not char:
            print("Empty file")
            sys.exit()

        # eat possible white spaces or empty lines
        while self.symbol_endline_or_space(char):
            char = f.read(1)

        comment = False
        if char == '%':
            comment = True

        char1 = f.read(1)
        
        while char1:
            l0 = ""
            while char1 and not(((char == '.' and not comment) and self.symbol_endline_or_space(char1)) or (comment and self.endline_symbol(char1))):
                # look for a . followed by \n
                l0 = l0 + char
                char = char1
                char1 = f.read(1)
            # new line
            l0 = l0 + char
            if not comment:
                l0 = l0.replace('\n', '').replace('\r', '')
                if "not " in l0: # to handle not fact, and avoid removing spaces, important space after not
                    l0 = l0.split("not")
                    l1 = ""
                    for el in l0:
                        el = el.replace(' ', '')
                        l1 = l1 + el + " not "
                    l1 = l1[:-4]  # remove last not
                elif l0.startswith('abducible'):
                    l0 = l0.split('abducible')
                    for i in range(1,len(l0)):
                        l0[i] = l0[i].replace(' ', '')
                    l1 = "abducible"
                    for el in range(1,len(l0)):
                        l1 = l1 + ' ' + l0[i]
                    # print(l1)
                else:
                    l1 = l0.replace(' ','')

                if l0[0].startswith('not_'):
                    utils.print_waring("The head of a clause that starts with not_ is not suggested.")
                    utils.print_waring("Hou should change its name. If not, you may get a wrong probability range")

                # hack to handle something like: 0.5::a % comment, to remove
                # the part after the %
                percent = l1.find('%')
                if percent != -1:
                    l1 = l1[:percent]

                self.lines_original.append(l1)
            char = char1
            # eat white spaces or empty lines
            char1 = f.read(1)
            while self.symbol_endline_or_space(char1):
                char1 = f.read(1)
            if char1 == '%':
                comment = True
            else:
                comment = False
            # print(char)
            # print(char1)
        f.close()
        self.parse_program()

    def parse_program(self) -> bool:
        n_probabilistic_facts = 0
        gen = generator.Generator()
        for line in self.lines_original:
            self.check_reserved(line)
            if "::" in line and not line.startswith('%'):
                if ':-' in line:
                    utils.print_error_and_exit("Probabilistic clauses are not supported\n" + line)
                if ';' in line:
                    utils.print_error_and_exit(
                        "Disjunction is not yet supported in probabilistic facts\nplease rewrite it as single fact.\nExample: 0.6::a;0.2::b. can be written as\n0.6::a. 0.5::b. where 0.5=0.2/(1 - 0.6)")
                # line with probability value
                probability, fact = self.check_consistent_prob_fact(line)

                self.add_probabilistic_fact(fact,probability)

                # clauses = gen.generate_clauses_for_facts(fact,probability,self.precision)

                # self.lines_prob.append(clauses)

                n_probabilistic_facts = n_probabilistic_facts + 1
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
                for el in expanded_conditional:
                    self.lines_prob.append([el])
            elif line.startswith("abducible"):
                _, abducible = gen.generate_clauses_for_abducibles(line, 0)
                # self.lines_prob.append(clauses)
                # self.abducibles.append(abducible)
                self.abducibles.append(abducible)
            elif self.is_number(line.split(':-')[0]):
                # probabilistic IC p:- body.
                # print("prob ic")
                # generate the probabilistic fact
                new_line = line.split(':-')[0] + "::icf" + str(self.n_probabilistic_ics) + "."
                probability, fact = self.check_consistent_prob_fact(new_line)
                self.add_probabilistic_fact(fact, probability)
                new_clause = "ic" + str(self.n_probabilistic_ics) + ":- " + line.split(':-')[1]
                self.lines_prob.append([new_clause])

                new_ic_0 = ":- icf" + str(self.n_probabilistic_ics) + ", ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append([new_ic_0])

                new_ic_1 = ":- not icf" + str(self.n_probabilistic_ics) + ", not ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append([new_ic_1])

                self.n_probabilistic_ics = self.n_probabilistic_ics + 1
                
            else:
                if not line.startswith("#show"):
                    self.lines_prob.append([line])
        if self.query is None:
            utils.print_error_and_exit("Missing query")
            
        self.lines_prob = [item for sublist in self.lines_prob for item in sublist]

        for fact in self.probabilistic_facts:
            clauses = gen.generate_clauses_for_facts(
                fact, self.probabilistic_facts[fact]/(10**self.precision), self.precision)
            for c in clauses:
                self.lines_prob.append(c)

        i = 0
        for abd in self.abducibles:
            # kind of hack, refactor generate_clauses_for abducibles TODO
            clauses, _ = gen.generate_clauses_for_abducibles("abducible " + abd + ".", i)
            i = i + 1
            for c in clauses:
                self.lines_prob.append(c)

        # for a in self.lines_prob:
        #     print(a)
        # sys.exit()
        return True

    # dummy check for reserved facts
    def check_reserved(self, line : str) -> None:
        if line.startswith('q:-'):
            utils.print_error_and_exit("q is a reserved fact")
        elif line.startswith('nq:-'):
            utils.print_error_and_exit("nq is a reserved fact")
        elif line.startswith('e:-'):
            utils.print_error_and_exit("e is a reserved fact")
        elif line.startswith('ne:-'):
            utils.print_error_and_exit("ne is a reserved fact")

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
    def get_content_to_compute_minimal_set_facts(self) -> list:
        if self.evidence is None:
            prog = self.lines_prob + [":- not " + self.query + "."]
        else:
            prog = self.lines_prob + [":- not " + self.evidence + "."]
        
        return prog

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
    def get_asp_program(self) -> str:
        self.lines_prob.append("q:- " + self.query + ".")
        self.lines_prob.append("#show q/0.")
        self.lines_prob.append("nq:- not " + self.query + ".")
        self.lines_prob.append("#show nq/0.")

        if self.evidence is not None:
            self.lines_prob.append("e:- " + self.evidence + ".")
            self.lines_prob.append("#show e/0.")
            self.lines_prob.append("ne:- not " + self.evidence + ".")
            self.lines_prob.append("#show ne/0.")

        return self.lines_prob

    def get_parsed_file(self) -> str:
        return self.lines_prob

    def has_evidence(self) -> bool:
        return self.evidence != None

    def error_prob_fact_twice(self, key : str, prob : float) -> None:
        print("Probabilistic fact " + key + " already defined with")
        print("probability " + str(self.probabilistic_facts[key]/(10**self.precision)) + ".")
        print("Trying to replace it with probability " + str(prob) + ".")

    # adds the current probabilistic fact and its probability in the 
    # list of probabilistic facts
    def add_probabilistic_fact(self, term : str, prob : float) -> None:
        key = term.split('.')[0]
        if key in self.probabilistic_facts:
            self.error_prob_fact_twice(key, prob)
            sys.exit()
        self.probabilistic_facts[key] = int(prob*(10**self.precision))

    '''
    string representation of the current class
    '''
    def __repr__(self) -> str:
        return "filename: " + self.filename + "\n" + \
        "precision: " + str(self.precision) + "\n" + \
        "query: " + str(self.query) + "\n" + \
        (("evidence: " + str(self.evidence) + "\n") if self.evidence != None else "") + \
        "probabilistic facts:\n" + str([str(x) + " " + str(y) for x, y in self.probabilistic_facts.items()]) + "\n" + \
        "n probabilistic facts:\n" + str(self.probabilistic_facts()) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "probabilities file:\n" + str(self.lines_prob) + "\n" + \
        (("abducibles: " + str(self.abducibles))  if self.abducibles != None else "")
