import argparse
import math
import networkx
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

    # command_parser.add_argument(
    #     "--map",
    #     help="Percentage of query atoms",
    #     type=float,
    #     default=0
    # )

    command_parser.add_argument(
        "--prob",
        help="Set the probability value (default random)",
        type=float,
        default=-1
    )

    command_parser.add_argument(
        "--solver",
        help="Generate the version for the selected solver",
        choices=["aspmc", "cplint", "PASTA", "ASP", "problog"],
        required=True
    )
    
    command_parser.add_argument(
        "--type",
        help="Type of graph",
        choices=["complete", "barabasi", "erdos"],
        default="complete"
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

def print_query_atoms(args : argparse.Namespace, g : 'networkx.Graph'):
    """
    Prints query atoms and probabilistic facts.
    Convoluted way but keeps the same probability among different
    runs with different map values.
    """
    prob_atoms : 'list[float]' = []
    for _ in range(len(g.edges)):
        prob = args.prob if (args.prob > 0 and args.prob < 1) else get_random_float()
        while prob < 0.001 or prob > 0.999:
            prob = get_random_float()
        prob_atoms.append(prob)

    for i in range(args.n):
        print(f"node({i}).")
    
    for e, p in zip(g.edges,prob_atoms):
        f = f"edge({e[0]},{e[1]})"
        if args.solver == "aspmc":
            print(f"{p}::{f}.")
            print(f"query({f}).")
        elif args.solver == "cplint":
            print(f"map_query {p}::{f}.")
        elif args.solver == "PASTA":
            print(f"map {p}::{f}.")
        elif args.solver == "problog":
            print(f"{p}::{f}.")
        elif args.solver == "ASP":
            print(f"% {p}")
            print("{" + f + "}.")
            lp = int(math.log(p)*1000)
            l_not_p = int(math.log(1 - p)*1000)
            print(f"log({lp},{f}) :- {f}.")
            print(f"log({l_not_p},{f}) :- not {f}.")
            # log(-510,edge_1_2) :- edge(1,2).
            # log(-916,edge_1_2) :- not edge(1,2).
            # log(-2302,edge_1_3) :- edge(1,3).
            # log(-105,edge_1_3) :- not edge(1,3).
            # print("TODO")
            # print(f"{f}.")


    if args.solver == "aspmc":
        print("evidence(qr).")
    elif args.solver == "problog":
        print("evidence(qr, true).")
    elif args.solver == "ASP":
        print(":- not qr.")
        print(":~ log(W,A). [-W,A]") 
    elif args.solver == "cplint":
        print(":- end_lpad.")

def main():
    """
    Main.
    """
    args = parse_args()
    print(f"% {args}")
    random.seed(args.seed)

    if args.solver == "cplint":
        print(":- use_module(library(pita)).")
        print(":- pita.")
        print(":- begin_lpad.")

    
    print("reaches(X,Y) :- edge(X,Y), node(X), node(Y).")
    print("reaches(X,Y) :- edge(X,Z), reaches(Z,Y), node(X), node(Y), node(Z).")
    print(f"qr:- reaches(0,{args.n - 1}).")
    
    if args.type == "barabasi":
        g = networkx.barabasi_albert_graph(args.n, 5)
    elif args.type == "erdos":
        g = networkx.erdos_renyi_graph(args.n, 0.5)
    else:
        g = networkx.complete_graph(args.n)
    
    # for edge in g.edges:
    #     print(f"edge({edge[0]},{edge[1]}).")
    
    print_query_atoms(args, g)


if __name__ == "__main__":
    main()