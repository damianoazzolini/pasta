import argparse
import copy
import random

def parse_args():
    """
    Arguments parser.
    """
    command_parser = argparse.ArgumentParser(
        description="Generate datasets with increasing rule numbers",
        # epilog="Example: ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    command_parser.add_argument(
        "--program",
        help="Type of programs to generate.",
        type=str,
        required=True,
        choices=["1","2", "3", "4"]
    )

    command_parser.add_argument(
        "-n",
        help="Number of probabilistic facts",
        type=int,
        required=True
    )
    
    command_parser.add_argument(
        "--map",
        help="Percentage of query atoms",
        type=float,
        default=-1
    )

    command_parser.add_argument(
        "--disj",
        help="Disjunction symbol",
        type=str,
        default=";",
        choices=[";","|"]
    )

    command_parser.add_argument(
        "--aspmc",
        help="Generate aspmc version with shifted negation",
        action="store_true"
    )
    
    command_parser.add_argument(
        "--prob",
        help="Set the probability value (default random)",
        type=float,
        default=-1
    )

    command_parser.add_argument(
        "--seed",
        help="Seed for random numbers generator",
        type=int,
        default=42
    )

    return command_parser.parse_args()

def get_random_float() -> float:
    """
    Get random float truncated at 3 decimals.
    """
    return float(str(random.random())[:5])

def print_query_atoms(args : argparse.Namespace):
    """
    Prints query atoms and probabilistic facts.
    Convoluted way but keeps the same probaiblity among different
    runs with different map values.
    """
    prob_atoms : 'list[float]' = []
    for _ in range(args.n):
        prob = args.prob if (args.prob > 0 and args.prob < 1) else get_random_float()
        prob_atoms.append(prob)
        
    selected_query_atoms = []
    if args.map >= 0:
        n_map = int(args.n * args.map)
        selected_query_atoms = random.sample(range(args.n), n_map)

    for idx, p in enumerate(prob_atoms):
        if idx in selected_query_atoms:
            if args.aspmc:
                print(f"{p}::a{idx}.")
                print(f"query(a{idx}).")
            else:
                print(f"map {p}::a{idx}.")
        else:
            print(f"{p}::a{idx}.")
    
    if args.aspmc:
        print("evidence(qr).")

def generate_first_type_programs(args : argparse.Namespace):
    """
    # generate rules, where the negation is random
    qr0 ; nqr0 :- not a0.
    qr0 :- not a1.
    qr2 ; nqr2 :-  a2.
    qr2 :-  a3.
    ... 
    """

    print_query_atoms(args)

    for i in range(0,args.n):
        if args.aspmc:
            prefix = "" if random.random() > 0.5 else "\+" # type: ignore
        else:
            prefix = "" if random.random() > 0.5 else "not"
        if i % 2 == 0:
            if args.aspmc:
                print(f"qr{i} :- {prefix} a{i}, \+ nqr{i}.") # type: ignore
                print(f"nqr{i} :- {prefix} a{i}, \+ qr{i}.") # type: ignore
            else:
                print(f"qr{i} {args.disj} nqr{i} :- {prefix} a{i}.")
        else:
            print(f"qr{i-1} :- {prefix} a{i}.")

    # for i in range(0,args.n,2):
    #     print(f"qr:- qr{i}.")
    b = ','.join([f"qr{i}" for i in range(0, args.n - args.n % 2, 2)])
    print(f"qr:- {b}.")
    
    print("% query: qr")
    
    
def generate_second_type_programs(args : argparse.Namespace):
    """
    # generate rules, where the negation is random
    qr0 ; nqr0:- a0.
    qr1 :- nqr0,  a1.
    qr2 ; nqr2 :- nqr1, not a2.
    ... 
    """
    print_query_atoms(args)
    
    if args.aspmc:
        prefix_pf = "" if random.random() > 0.5 else "\+" # type: ignore
    else:
        prefix_pf = "" if random.random() > 0.5 else "not"
    
    if args.aspmc:
        print(f"qr0:- {prefix_pf} a0, \+ nqr0.") # type: ignore
        print(f"nqr0:- {prefix_pf} a0, \+ qr0.") # type: ignore
    else:
        print(f"qr0 {args.disj} nqr0:- {prefix_pf} a0.")
    
    for i in range(1,args.n):
        prefix = "" if random.random() > 0.5 else "n"
        if args.aspmc:
            prefix_pf = "" if random.random() > 0.5 else "\+" # type: ignore
        else:
            prefix_pf = "" if random.random() > 0.5 else "not"
        if i % 2 == 0:
            if args.aspmc:
                print(f"qr{i} :- qr{i-1}, {prefix_pf} a{i}, \+ nqr{i}.") # type: ignore
                print(f"nqr{i} :- qr{i-1}, {prefix_pf} a{i}, \+ qr{i}.") # type: ignore
            else:
                print(f"qr{i} {args.disj} nqr{i} :- qr{i-1}, {prefix_pf} a{i}.")
        else:
            if i == 1:
                print(f"qr{i} :- nqr{i-1}, {prefix_pf} a{i}.")
            else:
                print(f"qr{i} :- {prefix}qr{i-1}, {prefix_pf} a{i}.")

    for i in range(0,args.n):
        print(f"qr:- qr{i}.")
    # b = ','.join([f"qr{i}" for i in range(0, args.n)])
    # print(f"qr:- {b}.")
    
    print("% query: qr")


def generate_third_type_programs(args : argparse.Namespace):
    """
    # programs of the form
    qr0 :- not a0.
    qr0;qr1 :- not a1.
    qr0;qr1;qr2 :-  a2.
    ... 
    """
    print_query_atoms(args)
    
    # random heads    
    # for i in range(0,args.n):
    #     h = [f"qr{i}" for i in range(0,args.n) if random.random() > 0.5]
    #     if len(h) == 0:
    #         # flip to have at least one true
    #         idx = random.randint(0,args.n-1)
    #         h = [f"qr{idx}"]
    #     hh = ";".join(h)
    #     prefix_pf = "" if random.random() > 0.5 else "not"
    #     print(f"{hh} :- {prefix_pf} a{i}.")
    # print("qr0:- a0.")
    for i in range(0, args.n):
        h = [f"qr{i}" for i in range(0,i+1)]
        
        if args.aspmc:
            prefix_pf = "" if random.random() > 0.5 else "\+" # type: ignore
            # shift the negation
            for idx, head in enumerate(h):
                h1 = copy.deepcopy(h)
                h1.pop(idx)
                if len(h1) > 0:
                    body = ",".join([f"\+ {at}" for at in h1]) # type:ignore
                    print(f"{head} :- {prefix_pf} a{i}, {body}.")
                else:
                    print(f"{head} :- {prefix_pf} a{i}.")
        else:
            hh = f"{args.disj}".join(h)
            prefix_pf = "" if random.random() > 0.5 else "not"
            print(f"{hh} :- {prefix_pf} a{i}.")
        
    for i in range(0,args.n):
        # h = [f"qr{i}" for i in range(0,args.n) if random.random() > 0.5]
        # hh = ",".join(h)
        # if random.random() > 0.5:
        print(f"qr:- qr{i}.")
    # b = ','.join([f"qr{i}" for i in range(0, args.n)])
    # print(f"qr:- {b}.")
    
    # print("% query: qr")
    
def generate_fourth_type_programs(args : argparse.Namespace):
    """
    Generate programs of the form
    qr ; nqr :- a0, not a1, not a2, ...
    qr :- a1, not a3, ...
    """
    
    def generate_body():
        b : 'list[str]' = []
        for i in range(0, args.n):
            if args.aspmc:
                prefix = "" if random.random() > 0.5 else "\+" # type: ignore
            else:
                prefix = "" if random.random() > 0.5 else "not"
            at = f"{prefix} a{i}"
            if random.random() > 0.5:
                b.append(at)
            
        return ','.join(b)
        
    ###### body
    print_query_atoms(args)
    
    bb = generate_body()
    if args.aspmc:
        print(f"qr :- {bb}, \+ nqr.") # type: ignore
        print(f"nqr :- {bb}, \+ qr.") # type: ignore
    else:
        print(f"qr {args.disj} nqr :- {bb}.")
    bb = generate_body()
    print(f"qr :- {bb}.")
    

def main():
    """
    Main function.
    """
    args = parse_args()
    print(f"% {args}")
    
    random.seed(args.seed)

    if args.program == "1":
        generate_first_type_programs(args)
    elif args.program == "2":
        generate_second_type_programs(args)
    elif args.program == "3":
        generate_third_type_programs(args)
    elif args.program == "4":
        generate_fourth_type_programs(args)

if __name__ == "__main__":
    main()