import time
import sys
from typing import Union

# local
import pasp_parser
import asp_interface
from utilities import parse_command_line

class Paspsp:
    def __init__(self, filename : str, query : str, evidence : str , precision=3, verbose=False, pedantic=False) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.precision = precision
        self.verbose = verbose
        self.pedantic = pedantic
    
    def solve(self) -> Union[float,float]:
        start_time = time.time()
        parser = pasp_parser.PaspParser(self.filename, self.precision, self.query, self.evidence)
        parser.parse()

        if verbose:
            print("Parsed program")

        content_find_minimal_set = parser.get_content_to_compute_minimal_prob_facts()

        asp_program = parser.get_asp_program()

        interface = asp_interface.AspInterface(content_find_minimal_set, evidence, asp_program, parser.get_dict_prob_facts(), self.precision)

        exec_time = interface.get_minimal_set_probabilistic_facts()

        if verbose:
            print("Computed cautious consequences in %s seconds" % (exec_time))
            if pedantic:
                print("--- Minimal set of probabilistic facts ---")
                print(interface.get_cautious_consequences())
                print("---")

        if pedantic:
            print("--- Asp program ---")
            interface.print_asp_program()
            print("---")
            print("--- Program to find minimal sets ---")
            for e in content_find_minimal_set:
                print(e)
            print("---")

        computed_models, n_worlds, grounding_time, computation_time, world_analysis_time = interface.compute_probabilities()
        end_time = time.time() - start_time

        if verbose:
            print("Computed models: " + str(computed_models))
            print("Considered worlds: " + str(n_worlds))
            print("Grounding time (s): " + str(grounding_time))
            print("Probability computation time (s): " + str(computation_time))
            print("World analysis time (s): " + str(world_analysis_time))
            print("Total time (s): " + str(end_time))

        uq = float(str(interface.get_upper_probability_query())[:8])
        lq = float(str(interface.get_lower_probability_query())[:8])

        if (lq > uq) or lq > 1 or uq > 1:
            print("Error in computing probabilities")
            print("Lower: " + str(lq))
            print("Upper: " + str(uq))
            sys.exit()

        return lq, uq
     

if __name__ == "__main__":
    verbose,pedantic,filename,precision,query,evidence = parse_command_line(sys.argv)

    pasp_solver = Paspsp(filename, query, evidence, precision, verbose, pedantic)
    lp,up = pasp_solver.solve()

    if lp == up:
        print("Lower probability == upper probability for the query " + query + ": " + str(lp))
    else:
        if query is None:
            print("Lower probability for the query: " + str(lp))
            print("Upper probability for the query: " + str(up))
        else:
            print("Lower probability for the query " + query + ": " + str(lp))
            print("Upper probability for the query " + query + ": " + str(up))
