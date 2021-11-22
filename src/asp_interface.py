import clingo
from typing import Union
import time

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
    def __init__(self,program_minimal_set : list, asp_program : list) -> None:
        self.cautious_consequences = ""
        self.program_minimal_set = program_minimal_set
        self.asp_program = asp_program
        self.lower_probability = 0
        self.upper_probability = 0

    def get_cautious_consequences(self) -> str:
        return self.cautious_consequences

    def get_lower_probability(self) -> str:
        return str(self.lower_probability)
    
    def get_upper_probability(self) -> str:
        return str(self.upper_probability)
    
    '''
    Parameters:
        - None
    Return:
        - str
    Behavior:
        compute the minimal set of probabilistic facts
        needed to make the query true.
        Cautious consequences
        clingo <filename> -e cautious
    '''
    def get_minimal_set_probabilistic_facts(self) -> float:
        ctl = clingo.Control(["--enum-mode=cautious"])
        for clause in self.program_minimal_set:
            ctl.add('base',[],clause)
        ctl.ground([("base", [])])
        start_time = time.time()

        cautious = ""
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                cautious = str(m) # i need only the last one
            handle.get()
        clingo_time = time.time() - start_time
        self.cautious_consequences = cautious

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
    def compute_probabilities(self) -> Union[int,float,float]:
        ctl = clingo.Control(["0","--project"])
        for clause in self.asp_program:
            ctl.add('base',[],clause)
        
        start_time = time.time()
        ctl.ground([("base", [])])
        grounding_time = time.time() - start_time

        n_models = 0
        start_time = time.time()
        model_handler = models_handler.ModelsHandler()

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                model_handler.add_value(str(m))
                n_models = n_models + 1
            handle.get()
        computation_time = time.time() - start_time

        return n_models,grounding_time,computation_time
