import time
import sys
from typing import Union

import argparse

# local
import pasta_parser
import asp_interface

profilation = False

examples_string_exact = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\""
examples_string_exact_evidence = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\""
examples_string_approximate = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --approximate"
examples_string_approximate_rej = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\" --rejection"

examples_strings = "Examples:\n\n" + examples_string_exact + "\n\n" + examples_string_exact_evidence + "\n\n" + examples_string_approximate + "\n\n" + examples_string_approximate_rej

pasta_description = "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"

class Pasta:
    def __init__(self, filename : str, query : str, evidence : str , precision=3, verbose=False, pedantic=False, samples=1000) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.precision = precision
        self.verbose = verbose
        self.pedantic = pedantic
        if pedantic is True:
            self.verbose = True
        self.samples = samples

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

    def approximate_solve(self, args, from_string : str = None) -> Union[float,float]:
        # start_time = time.time()
        program_parser = pasta_parser.PastaParser(self.filename, self.precision, self.query, self.evidence)
        program_parser.parse_approx(from_string)
        asp_program = program_parser.get_asp_program()

        interface = asp_interface.AspInterface([], self.evidence, asp_program, program_parser.probabilistic_facts, len(program_parser.abducibles), self.precision, self.verbose, self.pedantic,self.samples,program_parser.probabilistic_facts)

        if self.evidence is None:
            lp, up = interface.sample_query()
        else:
            if args.rejection is True:
                lp, up = interface.rejection_sampling()
            elif args.mh is True:
                lp, up = interface.mh_sampling()
            elif args.gibbs is True:
                lp, up = interface.gibbs_sampling(args.block)

        # end_time = time.time() - start_time

        return str(lp), str(up)

    def solve(self, from_string : str = None) -> Union[float,float,list]:
        start_time = time.time()
        program_parser = pasta_parser.PastaParser(self.filename, self.precision, self.query, self.evidence)
        program_parser.parse(from_string)

        if self.verbose:
            print("Parsed program")

        content_find_minimal_set = program_parser.get_content_to_compute_minimal_set_facts()

        asp_program = program_parser.get_asp_program()

        interface = asp_interface.AspInterface(content_find_minimal_set, self.evidence, asp_program, program_parser.probabilistic_facts, len(program_parser.abducibles), self.precision, self.verbose, self.pedantic)

        # interface.print_asp_program()
        
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
            if profilation:
                import cProfile
                import pstats
                import io

                pr = cProfile.Profile()
                pr.enable()
                # https://docs.python.org/3/library/profile.html#pstats.Stats
            
            interface.abduction()

            if profilation:
                pr.disable()
                s = io.StringIO()
                from pstats import SortKey
                sortby = SortKey.CUMULATIVE
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                fp = open('profilation.txt','w')
                fp.write(s.getvalue())
                fp.close()

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
            uq = interface.upper_probability_query
            lq = interface.lower_probability_query

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
            print("Lower probability == upper probability for the query " + query + ": " + lp)
        else:
            print("Lower probability for the query " + query + ": " + lp)
            print("Upper probability for the query " + query + ": " + up)

if __name__ == "__main__":
    command_parser = argparse.ArgumentParser(
        description=pasta_description, epilog=examples_strings)
    command_parser.add_argument("filename",help="Program to analyse",type=str)
    command_parser.add_argument("-q","--query", help="Query", type=str)
    command_parser.add_argument("-e","--evidence", help="Evidence", type=str)
    command_parser.add_argument("-v","--verbose", help="Verbose mode, default: false", action="store_true")
    command_parser.add_argument("--pedantic", help="Pedantic mode, default: false", action="store_true")
    command_parser.add_argument("-p", "--precision", help="Precision, default 3", type=int, default=3)
    command_parser.add_argument("--approximate", help="Compute approximate probability", action="store_true")
    command_parser.add_argument("--samples", help="Number of samples, default 1000", type=int, default=1000)
    command_parser.add_argument("--mh", help="Use Metropolis Hastings sampling", action="store_true", default=False)
    command_parser.add_argument("--gibbs", help="Use Gibbs Sampling sampling", action="store_true", default=False)
    command_parser.add_argument("--block", help="Set the block value for Gibbs sampling", type=int, default=1)
    command_parser.add_argument("--rejection", help="Use rejection Sampling sampling", action="store_true", default=False)
    
    args = command_parser.parse_args()

    pasta_solver = Pasta(args.filename, args.query, args.evidence, args.precision, args.verbose, args.pedantic, args.samples)
    
    if args.approximate is False:
        lp, up, abd_explanations = pasta_solver.solve()
    else:
        lp, up = pasta_solver.approximate_solve(args)
        abd_explanations = None

    if lp != None:
        print_prob(lp,up,args.query)
    if abd_explanations is not None:
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
                if len(ls[i]) > 0:
                    if ls[i].issubset(ls[j]):
                        ls[j] = ''

        abd_explanations = ls
        print("Abductive explanations " + str(sum(1 for ex in abd_explanations if len(ex) > 0)))
        index = 0
        for el in abd_explanations:
            if len(el) > 0:
                print("Explanation " + str(index))
                index = index + 1
                print(sorted(el))
