'''
Class defining a parser for a PASP program.
'''
import math
import os
import sys

# local
import utilities

class PaspParser:
    '''
    Parameters:
        - filename: name of the file to read
        - query: query to answer
        - precision: multiplier for the log probabilities
        - lines_original: lines from the parsing of the original file
        - lines_log_prob: lines obtained by replacing probabilities 
          with log probabilities
    '''
    def __init__(self,filename,precision=1000) -> None:
        self.filename = filename
        self.precision = precision
        self.lines_original = []
        self.lines_log_prob = []

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
        while utilities.endline_content(char):
            char = f.read(1)

        char1 = f.read(1)

        comment = False
        if char == '%':
            comment = True
        
        while char1:
            l0 = ""
            while char1 and not(((char == '.' and not comment) and utilities.endline_content(char1)) or (comment and utilities.endline_comment(char1))):
                # look for a . followed by \n
                l0 = l0 + char
                char = char1
                char1 = f.read(1)
            # new line
            l0 = l0 + char
            if not comment:
                self.lines_original.append(l0.replace('\n','').replace('\r','').replace(' ',''))
            char = char1
            # eat white spaces or empty lines
            char1 = f.read(1)
            while utilities.endline_content(char1):
                char1 = f.read(1)
            if char1 == '%':
                comment = True
            else:
                comment = False
            # print(char)
            # print(char1)
        
        self.insert_worlds_generator()

    '''
    Parameters:
        - none
    Return:
        - none
    Behavior:
        from P::f(1..n), creates:
        dom(1..n) -> domains
        0{v(I):dom(I)}n -> generator
        f(1,P1):- v(1).
        not_f(1,P0):- not v(1).
        ...
        f(n,P1):- v(n).
        not_f(n,P0):- not v(n).

        where P1 = -log(P)*precision
    '''
    def insert_worlds_generator(self) -> bool:
        # TODO: handle 0.5::f(1), 0.5::f(2)
        # i.e., probabilistic facts with the same functor, for the
        # domain generation (dom = dom + ...)
        # print(self.lines_original)
        for line in self.lines_original:
            if "::" in line and not line.startswith('%'):
                # line with probability value
                # for example: line = ['0.5', 'f(1).']
                probability, fact = utilities.check_consistent_prob_fact(line)
                # print(fact)
                # extract the value between brackets, 1 in the example
                arguments = utilities.extract_atom_between_brackets(fact)

                functor = utilities.get_functor(fact)

                # print(functor)
                # print(arguments)

                dom,args = utilities.generate_dom_fact(functor,arguments)
                self.lines_log_prob.append([dom])
                # print(dom)
                # print(args)

                generator, clauses = utilities.generate_generator(functor,args,arguments,probability,self.precision)

                # print(generator)
                # print(clauses)
                clauses.append(generator)
                self.lines_log_prob.append(clauses)
            else:
                self.lines_log_prob.append([line])
        
        # flatten the list, maybe try to avoid this
        self.lines_log_prob = [item for sublist in self.lines_log_prob for item in sublist]
        return True

    '''
    Parameters:
        - program: program stored as a list of strings
        - query: query to add in the program
    Return:
        - list of strings modified as explained below
    Behavior:
        adds a string with :- not query.
    '''
    def add_query_constraint(self,query) -> bool:
        self.lines_log_prob.append(query)
        return True

    '''
    Parameters:
        - self
        - filename
    Return:
        - None
    Behavior:
        dumps the content of the string self.log_probabilities_file on 
        the file named filename
    '''
    def create_log_probabilities_file(self,filename) -> None:
        pass
    
    '''
    string representation of the current class
    '''
    def __repr__(self) -> str:
        return "filename: " + self.filename + "\n" + \
        "precision: " + str(self.precision) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "log probabilities file:\n" + str(self.lines_log_prob)

        