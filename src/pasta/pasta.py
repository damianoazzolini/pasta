import time
import sys
from tkinter.ttk import Progressbar
from typing import Union

import argparse

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
    
    def solve(self) -> Union[float,float,list]:
        start_time = time.time()
        program_parser = pasta_parser.PastaParser(self.filename, self.precision, self.query, self.evidence)
        program_parser.parse()

        if self.verbose:
            print("Parsed program")

        content_find_minimal_set = program_parser.get_content_to_compute_minimal_set_facts()

        asp_program = program_parser.get_asp_program()

        interface = asp_interface.AspInterface(content_find_minimal_set, self.evidence, asp_program, program_parser.get_dict_prob_facts(), len(program_parser.abducibles), self.precision, self.verbose)

        exec_time = interface.get_minimal_set_facts()

        if self.verbose:
            print("Computed cautious consequences in %s seconds" % (exec_time))
            if self.pedantic:
                print("--- Minimal set of probabilistic facts ---")
                print(interface.cautious_consequences)
                print("---")

        if self.pedantic:
            print("--- Asp program ---")
            interface.print_asp_program()
            print("---")
            print("--- Program to find minimal sets ---")
            for e in content_find_minimal_set:
                print(e)
            print("---")

        if len(program_parser.abducibles) > 0:
            interface.abduction()
        else:
            interface.compute_probabilities()
        end_time = time.time() - start_time

        if self.verbose:
            print("Computed models: " + str(interface.computed_models))
            print("Considered worlds: " + str(interface.n_worlds))
            print("Grounding time (s): " + str(interface.grounding_time))
            print("Probability computation time (s): " + str(interface.computation_time))
            print("World analysis time (s): " + str(interface.world_analysis_time))
            print("Total time (s): " + str(end_time))

        # print(program_parser)

        if len(program_parser.probabilistic_facts) > 0:
            uq = interface.get_upper_probability_query()
            lq = interface.get_lower_probability_query()

            if (lq > uq) or lq > 1 or uq > 1:
                print("Error in computing probabilities")
                print("Lower: " + '{:.8f}'.format(lq))
                print("Upper: " + '{:.8f}'.format(uq))
                sys.exit()

        # print(lq)
        # print(uq)
        # print(interface.abductive_explanations)

        if len(program_parser.probabilistic_facts) == 0:
            return None, None, interface.abductive_explanations
        else:
            exp = interface.abductive_explanations if interface.n_abducibles > 0 else None
            return self.truncate_prob(lq)[:8], self.truncate_prob(uq)[:8], exp

def print_prob(lp : str, up : str, query : str) -> None:
    if query is None:
        if lp == up:
            print("Lower probability == upper probability for the query: " + lp)
        else:
            print("Lower probability for the query: " + lp)
            print("Upper probability for the query: " + up)
    else:
        if lp == up:
            print(
                "Lower probability == upper probability for the query " + args.query + ": " + lp)
        else:
            print("Lower probability for the query " + query + ": " + lp)
            print("Upper probability for the query " + query + ": " + up)

if __name__ == "__main__":
    command_parser = argparse.ArgumentParser()
    command_parser.add_argument("filename",help="Program to analyse",type=str)
    command_parser.add_argument("-q","--query", help="Query", type=str)
    command_parser.add_argument("-e","--evidence", help="Evidence", type=str)
    command_parser.add_argument("-v","--verbose", help="Verbose mode, default: false", action="store_true")
    command_parser.add_argument("--pedantic", help="Pedantic mode, default: false", action="store_true")
    command_parser.add_argument("-p", "--precision", help="Precision, default 3", type=int, default=3)
    
    args = command_parser.parse_args()

    pasta_solver = Pasta(args.filename, args.query, args.evidence, args.precision, args.verbose, args.pedantic)
    
    lp, up, abd_explanations = pasta_solver.solve()

    print_prob(lp,up,args.query)
    if abd_explanations is not None:
        print("Abductive explanations ")
        # remove dominated
        ls = []
        for el in abd_explanations:
            s = set()
            for a in el:
                if a.startswith("abd"):
                    s.add(a[4:])
            ls.append(s)

        for i in range(0,len(ls)):
            for j in range(i+1,len(ls)):
                if ls[i].issubset(ls[j]):
                    ls[j] = ''

        abd_explanations = ls

        # print(abd_explanations)
        for i in range(0,len(abd_explanations)):
            if len(abd_explanations[i]) > 0:
                print("Explanation " + str(i))
                print(sorted(abd_explanations[i]))
