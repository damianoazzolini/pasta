import argparse
import random
import networkx
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
            print(f"map {p}::node({idx}).")
        else:
            print(f"{p}::node({idx}).")


def main():
    """
    Main.
    """
    args = parse_args()
    random.seed(args.seed)
    
    print("reaches(X,Y) :- edge(X,Y), node(X), node(Y).")
    print("reaches(X,Y) :- edge(X,Z), reaches(Z,Y), node(X), node(Y), node(Z).")
    print(f"qr:- reaches(0,{args.n - 1}).")
    
    g = networkx.complete_graph(args.n)
    
    for edge in g.edges:
        print(f"edge({edge[0]},{edge[1]}).")
    
    print_query_atoms(args)


if __name__ == "__main__":
    main()