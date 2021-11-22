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
        self.query = ""
        self.model_query_clause = ""
        self.model_not_query_clause = ""

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
        functors_list = [] # needed to store the extracted functors
        for line in self.lines_original:
            if "::" in line and not line.startswith('%'):
                # line with probability value
                # for example: line = ['0.5', 'f(1).']
                probability, fact = utilities.check_consistent_prob_fact(line)
                # extract the value between brackets, 1 in the example
                arguments = utilities.extract_atom_between_brackets(fact)

                functor = utilities.get_functor(fact)
                dom, args = utilities.generate_dom_fact(functor,arguments)
                
                self.lines_log_prob.append([dom])

                generator, clauses, functor = utilities.generate_generator(functor,args,arguments,probability,self.precision)

                # the variable functor is updated: now it contains the
                # functor for a range or the whole probabilistic fact for
                # a probabilistic fact
                # For example:
                # bird(1..2) -> functor = "bird"
                # bird(1) -> functor = "bird(1)"
                functors_list.append(functor)

                clauses.append(generator)
                self.lines_log_prob.append(clauses)
            else:
                self.lines_log_prob.append([line])
            
            # generate the model clause
            # Do here since i need to know how the number of probabilistic facts
        self.model_query_clause, self.model_not_query_clause = utilities.generate_model_clause(functors_list,self.query)
        
        # print("----- model clauses ")
        # print(model_clauses)

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
    def add_query(self, query : str) -> bool:
        if query.endswith('.'):
            query = query[:-1]
        self.query = query
        return True

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
    def get_content_to_compute_minimal_prob_facts(self) -> str:
        l1 = self.lines_log_prob
        l1.append(":- not " + self.query + ".")
        return l1
    
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
        res = self.lines_log_prob

        res.append(self.model_query_clause)
        res.append(self.model_not_query_clause)
        
        return res


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
    Parameters:
        - self
    Return:
        - str
    Behavior:
        returns the parsed file in a string
    '''
    def get_parsed_file(self) -> str:
        return self.lines_log_prob

    def get_n_prob_facts(self) -> int:
        return self.n_prob_facts

    '''
    string representation of the current class
    '''
    def __repr__(self) -> str:
        return "filename: " + self.filename + "\n" + \
        "precision: " + str(self.precision) + "\n" + \
        "query: " + str(self.query) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "log probabilities file:\n" + str(self.lines_log_prob) + "\n" \
        "model query:\n" + str(self.model_query_clause) + "\n" \
        "model not query:\n" + str(self.model_not_query_clause)

        