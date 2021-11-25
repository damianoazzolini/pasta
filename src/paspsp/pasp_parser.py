'''
Class defining a parser for a PASP program.
'''
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
    def __init__(self,filename,precision,query="") -> None:
        self.filename = filename
        self.precision = precision
        self.lines_original = []
        self.lines_log_prob = []
        self.query = query

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
        f.close()
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

        where P1 = -log(P)*(10**precision)
    '''
    def insert_worlds_generator(self) -> bool:
        # TODO: handle 0.5::f(1), 0.5::f(2)
        # i.e., probabilistic facts with the same functor, for the
        # domain generation (dom = dom + ...)
        # print(self.lines_original)
        n_probabilistic_facts = 0
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

                generator, clauses = utilities.generate_generator(functor,args,arguments,probability,self.precision)

                clauses.append(generator)
                self.lines_log_prob.append(clauses)
                n_probabilistic_facts = n_probabilistic_facts + 1
            elif line.startswith("query"):
                if line[-1] == ".":
                    self.query = line.split("query")[1][:-2][1:]
                else:
                    self.query = line.split("query")[1][:-1][1:]
            else:
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
        if self.query == "":
            print("Missing query")
            sys.exit()

        return self.lines_log_prob + [":- not " + self.query + "."]
    
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
        
        return self.lines_log_prob

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

        
