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
        - query: query to answer
        - precision: multiplier for the log probabilities
        - lines_original: lines from the parsing of the original file
        - lines_prob: lines obtained by replacing probabilities 
          with log probabilities
    '''

    def __init__(self, filename: str, precision: int, query = None, evidence = None, identify_useless_facts = False) -> None:
        self.filename = filename
        self.precision = precision
        self.query = query
        self.evidence = evidence
        # self.identify_useless_facts = False
        self.lines_original = []
        self.lines_prob = []
        self.probabilistic_facts = dict() # pairs [fact,prob]
        self.abducibles = []
        self.prolog_time = 0

    def get_n_prob_facts(self) -> int:
        return len(self.probabilistic_facts)

    def get_dict_prob_facts(self) -> dict:
        return self.probabilistic_facts

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
    Parameters:
        - verbose: default 0
    Returns:
        - list of strings representing the program
    Behavior:
        - parses the file and extract the lines
    '''
    def parse(self) -> None:
        if os.path.isfile(self.filename) == False:
            print("File " + self.filename + " not found")
            sys.exit()
        
        f = open(self.filename,"r")
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
            elif line.startswith("query"):
                # remove the "query" functor and handles whether the line
                # does not terminate with .
                # query(fly(1)) -> fly(1)
                if line[-1] == ".":
                    self.query = line.split("query")[1][:-2][1:]
                else:
                    self.query = line.split("query")[1][:-1][1:]
            elif line.startswith("evidence"):
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
            else:
                if not line.startswith("#show"):
                    self.lines_prob.append([line])
        if self.query is None:
            utils.print_error_and_exit("Missing query")
            
        self.lines_prob = [item for sublist in self.lines_prob for item in sublist]

        # if self.identify_useless_facts:
        #     start = timer()
        #     from pyswip import Prolog
        #     prolog = Prolog()
        #     prolog.consult(
        #         "/mnt/c/Users/damia/Desktop/Ricerca/Repos/pasta/src/pasta/atoms_extractor.pl")
        #     filename_pl = "temp.pl"
        #     f = open(filename_pl,"w")
        #     f.write(":-style_check(-discontiguous).\n")
        #     f.write(":-style_check(-singleton).\n\n")
        #     f.write(":- table " + self.query.split("(")[0] + "/" + str(self.query.count(',') + self.query.count('(')) + ".\n\n")
        #     n_ics = 0
        #     goal_index = 0
        #     for l in self.lines_prob:
        #         l1, n_ics, goal_index, ic_to_goal = generator.Generator.to_prolog(l,n_ics,goal_index)
        #         for el in l1:
        #             # print(el)
        #             # prolog.assertz(el)
        #             f.write(el + '\n')
        #         if ic_to_goal:
        #             # a = self.query + ":- newGoalFromICInserted" + str(n_ics - 1)
        #             # prolog.assertz(a)
        #             f.write(self.query + ":- newGoalFromICInserted" + str(n_ics - 1) + ".\n")
        #     p_facts_string = "["
        #     if len(self.probabilistic_facts) > 0:
        #         for pf in self.probabilistic_facts:
        #             # prolog.assertz(pf)
        #             f.write(pf + ".\n")
        #             p_facts_string = p_facts_string + pf + ","
        #         p_facts_string = p_facts_string[:-1] + "]"
        #     else:
        #         p_facts_string = p_facts_string + "]"
            
        #     abd_string = "["
        #     if len(self.abducibles) > 0:
        #         for abd in self.abducibles:
        #             abd_string = abd_string + abd + ","
        #             # prolog.assertz(abd)
        #             f.write(abd + ".\n")
        #         abd_string = abd_string[:-1] + "]"
        #     else:
        #         abd_string = abd_string + "]"
                        
        #     f.close()
            
        #     # call to prolog
        #     sys.exit()
            
        #     prolog.consult("temp.pl")
            
        #     query = "get_facts(" + self.query +"," + abd_string + "," + p_facts_string + ",A,F)"
        #     for sol in prolog.query(query):
        #         abd = sol["A"]
        #         prob_facts = sol["F"]

        #     end = timer()
        #     self.prolog_time = end - start
        #     # overwrite self.abducibles
        #     self.abducibles = [str(a) for a in abd]
        #     # remove prob facts from dict
        #     prob_facts = [str(a) for a in prob_facts]
        #     # print(prob_facts)
        #     # print(self.probabilistic_facts)
        #     for pf in list(self.probabilistic_facts):
        #         if pf not in prob_facts:
        #             del self.probabilistic_facts[pf]

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

        return True

    # dummy check for reserved facts
    def check_reserved(self, line : str) -> None:
        if line == 'q':
            utils.print_error_and_exit("q is a reserved fact")
        elif line == 'nq':
            utils.print_error_and_exit("nq is a reserved fact")
        elif line == 'e':
            utils.print_error_and_exit("e is a reserved fact")
        elif line == 'ne':
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
        "n probabilistic facts:\n" + str(self.get_n_prob_facts()) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "log probabilities file:\n" + str(self.lines_prob)
