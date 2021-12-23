import clingo
from typing import Union
import time
import sys

# local
import models_handler

class Context:
    def id(self, x):
        return x
    def seq(self, x, y):
        return [x, y]

def on_model(m):
    print (m)

class AspInterface:
    '''
    Parameters:
        - content: list with the program
    '''
    def __init__(self,program_minimal_set : list, evidence : list, asp_program : list, probabilistic_facts : dict, precision = 3) -> None:
        self.cautious_consequences = []
        self.program_minimal_set = program_minimal_set
        self.asp_program = asp_program
        self.lower_probability_query = 0
        self.upper_probability_query = 0
        self.upper_probability_evidence = 0
        self.lower_probability_evidence = 0
        self.precision = precision
        self.evidence = evidence
        self.probabilistic_facts = probabilistic_facts
        self.n_prob_facts = len(probabilistic_facts)

    def get_cautious_consequences(self) -> str:
        return self.cautious_consequences

    def get_lower_probability_query(self) -> float:
        return float(self.lower_probability_query)
    
    def get_upper_probability_query(self) -> float:
        return float(self.upper_probability_query)

    def get_lower_probability_evidence(self) -> float:
        return float(self.lower_probability_evidence)

    def get_upper_probability_evidence(self) -> float:
        return float(self.upper_probability_evidence)

    '''
    Parameters:
        - None
    Return:
        - str
    Behavior:
        compute the minimal set of probabilistic facts
        needed to make the query true. This operation is performed
        only if there is not evidence.
        Cautious consequences
        clingo <filename> -e cautious
    '''
    def get_minimal_set_probabilistic_facts(self) -> float:
        ctl = clingo.Control(["--enum-mode=cautious"])
        for clause in self.program_minimal_set:
            ctl.add('base',[],clause)

        ctl.ground([("base", [])])
        start_time = time.time()

        temp_cautious = []
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                # i need only the last one
                temp_cautious = str(m).split(' ')
            handle.get()

        for el in temp_cautious:
            if el != '' and (el.split(',')[-2] + ')' if el.count(',') > 0 else el.split('(')[0]) in self.probabilistic_facts:
                self.cautious_consequences.append(el)

        # sys.exit()
        clingo_time = time.time() - start_time

        return clingo_time

    '''
    Parameters:
        - None
    Return:
        - int: number of computed models
        - float: grounding time
        - float: computing probability time
    Behavior:
        compute the lower and upper bound for the query
        clingo 0 <filename> --project
    '''
    def compute_probabilities(self) -> Union[int,int,float,float,float]:
        ctl = clingo.Control(["0","--project"])
        for clause in self.asp_program:
            ctl.add('base',[],clause)

        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                ctl.add('base',[],":- not " + c + '.')
        
        start_time = time.time()
        ctl.ground([("base", [])])
        grounding_time = time.time() - start_time

        n_models = 0
        start_time = time.time()
        model_handler = models_handler.ModelsHandler(self.precision, self.n_prob_facts, self.evidence)

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                model_handler.add_value(str(m))
                n_models = n_models + 1
            handle.get()
        computation_time = time.time() - start_time

        # print(model_handler) # prints the models in world format

        start_time = time.time()
        self.lower_probability_query, self.upper_probability_query = model_handler.compute_lower_upper_probability()

        n_worlds = model_handler.get_number_worlds()

        world_analysis_time = time.time() - start_time

        return n_models,n_worlds,grounding_time,computation_time,world_analysis_time

    # prints the ASP program
    def print_asp_program(self) -> None:
        for el in self.asp_program:
            print(el)
        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                print(":- not " + c + '.')
