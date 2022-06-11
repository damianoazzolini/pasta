import sys

import argparse

# local
import pasta_parser
import asp_interface
import utils


examples_string_exact = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\""
examples_string_exact_evidence = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\""
examples_string_approximate = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --approximate"
examples_string_approximate_rej = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\" --rejection"

examples_strings = "Examples:\n\n" + examples_string_exact + "\n\n" + examples_string_exact_evidence + "\n\n" + examples_string_approximate + "\n\n" + examples_string_approximate_rej

pasta_description = "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"

class Pasta:
    def __init__(
        self, 
        filename : str, 
        query : str, 
        evidence : str , 
        verbose : bool = False, 
        pedantic : bool = False, 
        samples : int = 1000
        ) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.verbose = verbose
        self.pedantic = pedantic
        if pedantic is True:
            self.verbose = True
        self.samples = samples
        self.interface : asp_interface.AspInterface


    @staticmethod
    def remove_trailing_zeros(n : float) -> str:
        '''
        Removes trailing zeroes from floating point numbers
        Example: 0.25000 -> 0.25
        '''
        s = str('{:.8f}'.format(n))
        s0 = s.split('.')[0]
        s = s.split('.')[1]
        i = len(s)
        found = False
        while (i > 0) and (found is False):
            if int(s[i - 1]) != 0:
                found = True
            i = i - 1

        return s0 + "." + s[:i+1]


    @staticmethod
    def check_lp_up(lp : float, up : float) -> None:
        '''
        Checks whether lp =< up
        '''
        if (lp > up) or (int(lp*10e8) > 10e8) or (int(up*10e8) > 10e8):
            print("Error in computing probabilities")
            print("Lower: " + '{:.8f}'.format(lp))
            print("Upper: " + '{:.8f}'.format(up))
            sys.exit()


    def parameter_learning(self) -> None:
        pass


    def approximate_solve(self, args : argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Inference through sampling
        '''
        
        program_parser = pasta_parser.PastaParser(self.filename, self.query, self.evidence)
        program_parser.parse_approx(from_string)
        asp_program = program_parser.get_asp_program()

        self.interface = asp_interface.AspInterface([], self.evidence, asp_program, program_parser.probabilistic_facts, len(program_parser.abducibles), self.verbose, self.pedantic,self.samples,program_parser.probabilistic_facts)


        if self.evidence is None:
            lp, up = self.interface.sample_query()
        else:
            if args.rejection is True:
                lp, up = self.interface.rejection_sampling()
            elif args.mh is True:
                lp, up = self.interface.mh_sampling()
            elif args.gibbs is True:
                lp, up = self.interface.gibbs_sampling(args.block)
            else:
                lp = 0
                up = 0
                utils.print_error_and_exit("Sampling method found")
                
        return lp, up


    def setup_interface(self, from_string : str = "") -> None:
        '''
        Setup clingo interface
        '''
        program_parser = pasta_parser.PastaParser(
            self.filename, self.query, self.evidence)
        program_parser.parse(from_string)

        if self.verbose:
            print("Parsed program")

        content_find_minimal_set = program_parser.get_content_to_compute_minimal_set_facts()
        asp_program = program_parser.get_asp_program()

        self.interface = asp_interface.AspInterface(
            content_find_minimal_set, 
            self.evidence, 
            asp_program, 
            program_parser.probabilistic_facts,
            program_parser.abducibles, 
            self.verbose, 
            self.pedantic
        )

        exec_time = self.interface.get_minimal_set_facts()

        if self.verbose:
            print("Computed cautious consequences in %s seconds" % (exec_time))
            if self.pedantic:
                print("--- Minimal set of probabilistic facts ---")
                print(self.interface.cautious_consequences)
                print("---")

        if self.pedantic:
            print("--- Asp program ---")
            self.interface.print_asp_program()
            print("---")
            print("--- Program to find minimal sets ---")
            for e in content_find_minimal_set:
                print(e)
            print("---")
        

    def abduction(self, from_string: str = "") -> 'tuple[float,float,list[list[str]]]':
        '''
        Probabilistic and deterministic abduction
        '''
        self.setup_interface(from_string)
        self.interface.abduction()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        self.check_lp_up(lp,up)

        return lp, up, self.interface.abductive_explanations
    
    
    def inference(self, from_string : str = "") -> 'tuple[float,float]':
        '''
        Exact inference
        '''
        self.setup_interface(from_string)
        # self.interface.identify_useless_variables()
        self.interface.compute_probabilities()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        self.check_lp_up(lp,up)

        return lp, up


    @staticmethod
    def print_prob(lp : float, up : float) -> None:
        '''
        Print the probability values
        '''
        if lp == up:
            print(f"Lower probability == upper probability for the query: {lp}")
        else:
            print(f"Lower probability for the query: {lp}")
            print(f"Upper probability for the query: {up}")


    @staticmethod
    def remove_dominated_explanations(abd_exp : 'list[list[str]]') -> 'list[set[str]]':
        ls : list[set[str]] = [] 
        for exp in abd_exp:
            e : set[str] = set()
            for el in exp:
                if not el.startswith('not') and el != 'q':
                    if el.startswith('abd_'):
                        e.add(el[4:])
                    else:
                        e.add(el)
            ls.append(e)

        for i in range(0,len(ls)):
            for j in range(i+1,len(ls)):
                if len(ls[i]) > 0:
                    if ls[i].issubset(ls[j]):
                        ls[j] = set()  # type: ignore

        return ls


    @staticmethod
    def print_result_abduction(lp: float, up: float, abd_exp: 'list[list[str]]') -> None:
        abd_exp_no_dup = Pasta.remove_dominated_explanations(abd_exp)
        # abd_exp_no_dup = abd_exp
        if len(abd_exp_no_dup) > 0 and up != 0:
            Pasta.print_prob(lp,up)
        
        n_exp = sum(1 for ex in abd_exp_no_dup if len(ex) > 0)
        print(f"Abductive explanations: {n_exp}")

        index = 0
        for el in abd_exp_no_dup:
            if len(el) > 0:
                print(f"Explanation {index}")
                index = index + 1
                print(sorted(el))


if __name__ == "__main__":
    command_parser = argparse.ArgumentParser(
        description=pasta_description, epilog=examples_strings)
    command_parser.add_argument("filename",help="Program to analyse",type=str)
    command_parser.add_argument("-q","--query", help="Query", type=str)
    command_parser.add_argument("-e","--evidence", help="Evidence", type=str)
    command_parser.add_argument("-v","--verbose", help="Verbose mode, default: false", action="store_true")
    command_parser.add_argument("--pedantic", help="Pedantic mode, default: false", action="store_true")
    # command_parser.add_argument("--approximate", help="Compute approximate probability", action="store_true")
    # command_parser.add_argument("--samples", help="Number of samples, default 1000", type=int, default=1000)
    # command_parser.add_argument("--mh", help="Use Metropolis Hastings sampling", action="store_true", default=False)
    # command_parser.add_argument("--gibbs", help="Use Gibbs Sampling sampling", action="store_true", default=False)
    # command_parser.add_argument("--block", help="Set the block value for Gibbs sampling", type=int, default=1)
    # command_parser.add_argument("--rejection", help="Use rejection Sampling sampling", action="store_true", default=False)
    # command_parser.add_argument("-pl", help="Parameter learning", action="store_true", default=False)
    command_parser.add_argument("--abduction", help="Abduction", action="store_true", default=False)
    
    args = command_parser.parse_args()

    pasta_solver = Pasta(args.filename, args.query, args.evidence, args.verbose, args.pedantic, args.samples)
    
    args.approximate = False

    if args.abduction is True:
        lp, up, abd_explanations = pasta_solver.abduction()
        Pasta.print_result_abduction(lp, up, abd_explanations)
    elif args.approximate is True:
        lp, up = pasta_solver.approximate_solve(args)
        Pasta.print_prob(lp, up)
    elif args.pl is True:
        pasta_solver.parameter_learning()
    else:
        lp, up = pasta_solver.inference()
        Pasta.print_prob(lp, up)
