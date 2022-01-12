'''
Class defining a parser for a PASTA program.
'''
import os
import sys
from typing import Union
import re
# from typing import Dict, Union

import generator

class PastaParser:
    '''
    Parameters:
        - filename: name of the file to read
        - query: query to answer
        - precision: multiplier for the log probabilities
        - lines_original: lines from the parsing of the original file
        - lines_log_prob: lines obtained by replacing probabilities 
          with log probabilities
    '''
    def __init__(self, filename : str, precision : int ,query=None, evidence=None) -> None:
        self.filename = filename
        self.precision = precision
        self.lines_original = []
        self.lines_log_prob = []
        self.query = query
        self.evidence = evidence
        self.probabilistic_facts = dict() # pairs [fact,prob]

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

    # from fa(32) returns fa
    @staticmethod
    def get_functor(term: str) -> str:
        r = ""
        i = 0
        while i < len(term) and term[i] != '(':
            r = r + term[i]
            i = i + 1
        return r

    # can this be static?
    def check_consistent_prob_fact(self, line: str) -> Union[float, str]:
        if not line.endswith('.'):
            sys.exit("Missing final . in " + line)

        line = line.split("::")
        # for example: line = ['0.5', 'f(1..3).']
        if len(line) != 2:
            sys.exit("Error in parsing: " + str(line))

        if not self.is_number(line[0]):
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

    # from f(12) returns 12, does some basic checks
    # returns also True if range, false otherwise
    @staticmethod
    def extract_atom_between_brackets(fact: str) -> Union[list, bool]:
        val = re.findall(r'\(([^()]+)\)', fact)

        if len(val) == 1:
            if ".." in val[0]:
                return val[0].split(".."), True
            else:
                return val[0].split(','), False
        else:
            # fact not defined with a range
            return None, False

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
                else:
                    l1 = l0.replace(' ','')

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
        self.insert_worlds_generator()

    def insert_worlds_generator(self) -> bool:
        n_probabilistic_facts = 0
        gen = generator.Generator()
        for line in self.lines_original:
            self.check_reserved(line)
            if "::" in line and not line.startswith('%'):
                if ':-' in line:
                    print('Probabilistic clauses are not supported')
                    print("-> " + line)
                    sys.exit()
                if 'not_' in line:
                    print('Please define probabilistic facts without using not_')
                    print("-> " + line)
                    sys.exit()
                if ';' in line:
                    print('Disjunction is not yet supported in probabilistic facts')
                    print('please rewrite them as single facts.')
                    print('Example: 0.6::a;0.2::b. can be written as')
                    print('0.6::a. 0.5::b. where 0.5=0.2/(1 - 0.6)')
                # line with probability value
                probability, fact = self.check_consistent_prob_fact(line)
                arguments = self.extract_atom_between_brackets(fact)
                functor = self.get_functor(fact)

                self.add_probabilistic_fact(fact,probability)
                # print(self.probabilistic_facts)
                # sys.exit()

                dom, args = gen.generate_dom_fact(functor,arguments)

                self.lines_log_prob.append([dom])

                generat, clauses = gen.generate_generator(functor,args,arguments,probability,self.precision)

                clauses.append(generat)
                self.lines_log_prob.append(clauses)
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
                expanded_conditional = gen.expand_conditional(line)
                for el in expanded_conditional:
                    self.lines_log_prob.append([el])
            else:
                if not line.startswith("#show"):
                    self.lines_log_prob.append([line])
            
            # generate the model clause
            # Do here since i need to know how the number of probabilistic facts
        if n_probabilistic_facts == 0:
            print("This is not a probabilistic answer set program.")
            print("No probabilities detected.")
            print("Please specify at least one probabilistic fact with")
            print("prob::fact. For example: 0.5::a. states that a has")
            print("probability 0.5.")
            sys.exit()

        # flatten the list, maybe try to avoid this
        self.lines_log_prob = [item for sublist in self.lines_log_prob for item in sublist]

        return True

    # dummy check fo reserved facts
    def check_reserved(self, line : str) -> None:
        if line == 'q':
            print("q is a reserved fact")
            sys.exit()
        elif line == 'nq':
            print("nq is a reserved fact")
            sys.exit()
        elif line == 'e':
            print("e is a reserved fact")
            sys.exit()
        elif line == 'ne':
            print("ne is a reserved fact")
            sys.exit()


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
    def get_content_to_compute_minimal_prob_facts(self) -> list:
        if self.query is None:
            print("Missing query")
            sys.exit()
        
        if self.evidence is None:
            prog = self.lines_log_prob + [":- not " + self.query + "."]
        else:
            prog = self.lines_log_prob + [":- not " + self.evidence + "."]
        
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
        self.lines_log_prob.append("q:- " + self.query + ".")
        self.lines_log_prob.append("#show q/0.")
        self.lines_log_prob.append("nq:- not " + self.query + ".")
        self.lines_log_prob.append("#show nq/0.")

        if self.evidence is not None:
            self.lines_log_prob.append("e:- " + self.evidence + ".")
            self.lines_log_prob.append("#show e/0.")
            self.lines_log_prob.append("ne:- not " + self.evidence + ".")
            self.lines_log_prob.append("#show ne/0.")


        return self.lines_log_prob

    def get_parsed_file(self) -> str:
        return self.lines_log_prob

    def has_evidence(self) -> bool:
        return self.evidence != None

    def error_prob_fact_twice(self, key : str, prob : float) -> None:
        print("Probabilistic fact " + key + " already defined with")
        print("probability " + str(self.probabilistic_facts[key]/(10**self.precision)) + ".")
        print("Trying to replace it with probability " + str(prob) + ".")

    # adds the current probabilistic fact and its probability in the 
    # list of probabilistic facts. Also explodes the ranges
    def add_probabilistic_fact(self, term : str, prob : float) -> None:
        if ".." in term:
            line = term.split("(")
            functor = line[0]
            interval = line[1][:-1] # to remove )
            interval = interval.split("..")
            lb = int(interval[0])
            ub = int(interval[1])

            for i in range(lb, ub + 1):
                key = functor + "(" + str(i) + ")"
                if key in self.probabilistic_facts:
                    self.error_prob_fact_twice(key,prob)
                    sys.exit()
                self.probabilistic_facts[key] = int(prob*(10**self.precision))
        else:
            # split to remove the . (if present)
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
        "log probabilities file:\n" + str(self.lines_log_prob)
