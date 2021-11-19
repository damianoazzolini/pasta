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

        char1 = f.read(1)
        while char1:
            l0 = ""
            while char1 and not((char == '.') and (char1 == '\n' or char1 == '\r\n' or char1 == ' ')):
                # look for a . followed by \n
                l0 = l0 + char
                char = char1
                char1 = f.read(1)
            # new line
            l0 = l0 + char
            self.lines_original.append(l0.replace('\n','').replace('\r','').replace(' ',''))
            char = char1
            char1 = f.read(1)
        
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
    def insert_worlds_generator(self):
        # TODO: handle 0.5::f(1), 0.5::f(2)
        # i.e., probabilistic facts with the same functor, for the
        # domain generation (dom = dom + ...)
        print(self.lines_original)
        for line in self.lines_original:
            if "::" in line:
                # line with probability value
                # for example: line = ['0.5', 'f(1).']
                probability, fact = utilities.check_consistent_prob_fact(line)
                # print(fact)
                # extract the value between brackets, 1 in the example
                arguments = utilities.extract_atom_between_brackets(fact)

                functor = utilities.get_functor(fact)

                print(functor)
                print(arguments)

                dom = utilities.generate_dom_fact(functor,arguments)
                # TODO: return also args? needed below

                # generates the dom fact
                # dom_f(1..3). from f(1..3).
                # dom = "dom_" + functor + "("

                # args = ""
                # if arguments[1] is False: # single fact
                #     for a in arguments[0]:
                #         args = args + a + ","
                #     args = args[:-1] # remove last ,
                #     args = args + ")"
                # else: # range
                #     # args = args + arguments[0][0] + ".." + arguments[0][1] + ")"
                #     args = "I)"

                # if arguments[1] is False: # single fact
                #     dom_args = args
                # else: # range
                #     dom_args = arguments[0][0] + ".." + arguments[0][1] + ")"
                    

                # dom = dom + dom_args + "."

                print(dom)

                # generates the generator
                # 0{v_f_(I):dom_f(I)}3. from f(1..3)
                vt = "v_" + functor + "_(" + args

                generator = ""
                generator = generator + "0{"
                if arguments[1] is False:
                    number = 1
                else:
                    number = int(arguments[0][1]) - int(arguments[0][0]) + 1
                
                generator = generator + vt + ":dom_" + functor + "(" + args + "}" + str(number) + "."
                
                print(generator)

                # generate the two clauses for true and false
                # f = ""
                # nf = ""
                # if arguments[1] is False: # single fact
                #     f = functor + "(" + args + ":- " + vt + "."
                #     nf = "not_" + functor + "(" + args + ":- not " + vt + "."
                # else: # range
                #     for i in range(int(arguments[0][0]),int(arguments[0][1])):
                #         f = functor + "(" + args + ":- " + vt + "."
                #         nf = "not_" + functor + "(" + args + ":- not " + vt + "."


                



                    


    '''
    Parameters:
        - program: program stored as a list of strings
        - query: query to add in the program
    Return:
        - list of strings modified as explained below
    Behavior:
        adds a string with :- not query.
    '''
    def add_query_constraint(program,query) -> list:
        return program.append(query)
    
    '''
    string representation of the current class
    '''
    def __repr__(self) -> str:
        print("filename: " + self.filename)
        print("precision: " + str(self.precision))
        print("original file")
        print(self.lines_original)
        print("log probabilities file")
        print(self.lines_log_prob)

        