from os import stat
import time
import sys
from typing import Union

# local
import pasta_parser
import asp_interface

class Pasta:
    def __init__(self, filename : str, query : str, evidence : str , precision=3, verbose=False, pedantic=False) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.precision = precision
        self.verbose = verbose
        self.pedantic = pedantic

    @staticmethod
    def print_help() -> None:
        print("PASTA: Probabilistic Answer Set programming for STAtistical probabilities")
        # print("Compute lower and upper bound for a query in")
        # print("a probabilistic answer set program")
        print("pasta <program> [OPTIONS]")
        print("Example: pasta ../../examples/bird_4.lp -q=\"fly(1)\"")
        print("Example programs: see example folder.")
        print("Issues: https://github.com/damianoazzolini/pasta/issues")
        print("Available commands:")
        print("\t--query=,-q: specifies a query. Example: -q=\"fly(1)\".")
        print("\t\tIt can also be specified in the program by adding the line query(fly(1)).")
        print("\t--evidence=,-e: specifies a evidence. Example: -e=\"fly(1)\".")
        print("\t\tIt can also be specified in the program by adding the line evidence(fly(1)).")
        print("\t--verbose,-v: verbose mode. Default: off.")
        print("\t--pedantic: pedantic mode (more verbose than --verbose). Default: off.")
        print("\t--precision=,-p=: set the required precision. Example: --precision=3. Default = 3.")
        print("\t--help,-h: print this help page")
    
    @staticmethod
    def parse_command_line(args: str) -> Union[bool, bool, str, int, str, str]:
        verbose = False
        pedantic = False
        filename = ""
        precision = 3  # default value
        query = None
        evidence = None
        # for i in range(0,len(args)):
        i = 0
        while i < len(args):
            if args[i] == "--verbose" or args[i] == "-v":
                verbose = True
            elif args[i] == "--pedantic":
                verbose = True
                pedantic = True
            elif args[i].startswith("--precision=") or args[i].startswith("-p="):
                precision = int(args[i].split('=')[1])
            elif args[i] == "--help" or args[i] == "-h":
                Pasta.print_help()
                sys.exit()
            elif args[i].startswith("--query=") or args[i].startswith("-q="):
                query = args[i].split("=")[1].replace("\"", "")
            elif args[i].startswith("--evidence=") or args[i].startswith("-q="):
                evidence = args[i].split("=")[1].replace("\"", "")
            else:
                if i + 1 < len(args):
                    filename = args[i+1]
                    i = i + 1
            i = i + 1

        if filename == "":
            print("Missing filename")
            sys.exit()

        return verbose, pedantic, filename, precision, query, evidence

    @staticmethod
    def truncate_prob(s: float) -> str:
        s = str('{:.8f}'.format(s))
        s0 = s.split('.')[0]
        s = s.split('.')[1]
        i = len(s)
        found = False
        while (i > 0) and (found is False):
            if int(s[i - 1]) != 0:
                found = True
            i = i - 1

        return s0 + "." + s[:i+1]
    
    def solve(self) -> Union[float,float]:
        start_time = time.time()
        parser = pasta_parser.PastaParser(self.filename, self.precision, self.query, self.evidence)
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

        uq = interface.get_upper_probability_query()
        lq = interface.get_lower_probability_query()

        if (lq > uq) or lq > 1 or uq > 1:
            print("Error in computing probabilities")
            print("Lower: " + '{:.8f}'.format(lq))
            print("Upper: " + '{:.8f}'.format(uq))
            sys.exit()

        return self.truncate_prob(lq)[:8], self.truncate_prob(uq)[:8]
     

if __name__ == "__main__":
    verbose,pedantic,filename,precision,query,evidence = Pasta.parse_command_line(sys.argv)

    pasta_solver = Pasta(filename, query, evidence, precision, verbose, pedantic)
    lp,up = pasta_solver.solve()    

    if query is None:
        if lp == up:
            print("Lower probability == upper probability for the query: " + lp)
        else:
            print("Lower probability for the query: " + lp)
            print("Upper probability for the query: " + up)
    else:
        if lp == up:
            print("Lower probability == upper probability for the query " + query + ": " + lp)
        else:
            print("Lower probability for the query " + query + ": " + lp)
            print("Upper probability for the query " + query + ": " + up)
