import argparse
import random
import sys

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
        "-n",
        help="Number of probabilistic facts",
        type=int,
        required=True
    )

    command_parser.add_argument(
        "--map",
        help="Percentage of query atoms",
        type=float,
        default=0
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
    
    command_parser.add_argument(
        "--disj",
        help="Disjunction symbol",
        type=str,
        default=";",
        choices=[";","|"]
    )

    return command_parser.parse_args()


def get_random_float() -> float:
    """
    Get random float truncated at 3 decimals.
    """
    return float(str(random.random())[:5])


def main():
    """
    Main body.
    """
    args = parse_args()
    print(f"% {args}")
    if args.n < 1:
        print("n must be at least 2.")
        sys.exit()
    
    if args.map < 0 or args.map > 1:
        print("map must be between 0 or 1, or not specified")
        sys.exit()
    
    random.seed(args.seed)

    preamble = """
influences(X,Y):- infl(X,Y).
influences(X,Y):- infl(Y,X).
smokes(X):- stress(X), stress_fact(X).
smokes(X) :- influences(Y, X), smokes(Y).
asthma_rule(X):- smokes(X), asthma_fact(X).
asthma(X):- asthma_f(X).
asthma(X):- asthma_rule(X).
"""
    print(preamble)
    
    if args.aspmc:
        print("smokes(X) :- asthma(X), \+ nsmokes(X).")
        print("nsmokes(X) :- asthma(X), \+ smokes(X).")
    else:
        print(f"smokes(X) {args.disj} nsmokes(X) :- asthma(X).")

    n_prob_f = (5 * args.n) - 1

    # write the probabilistic facts
    prob_atoms : 'list[float]' = []
    for _ in range(n_prob_f):
        prob = args.prob if (args.prob > 0 and args.prob < 1) else get_random_float()
        prob_atoms.append(prob)
    
    selected_query_atoms = []
    if args.map >= 0:
        n_map = int(n_prob_f * args.map)
        selected_query_atoms = random.sample(range(n_prob_f), n_map)
    
    names = ["asthma_f(ID)","stress(ID)","stress_fact(ID)","asthma_fact(ID)","infl(1,ID)"]
    
    expanded_names : 'list[str]' = []
    for name in names:
        for i in range(1, args.n + 1):
            if name == "infl(1,ID)" and i == 1:
                pass
            else:
                expanded_names.append(name.replace("ID",str(i)))

    for prob, f, idx in zip(prob_atoms,expanded_names,range(0,n_prob_f)):
        if idx in selected_query_atoms:
            if args.aspmc:
                print(f"{prob}::{f}.")
                print(f"query({f}).")
            else:
                print(f"map {prob}::{f}.")
        else:
            print(f"{prob}::{f}.")

    
    for i in range(1, args.n + 1):
        print(f"qr:- smokes({i}).")
    
    if args.aspmc:
        print("evidence(qr).")

if __name__ == "__main__":
    main()