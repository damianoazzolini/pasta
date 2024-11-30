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
    if args.n < 7:
        print("n must be at least 7.")
        sys.exit()
    
    if args.map < 0 or args.map > 1:
        print("map must be between 0 or 1, or not specified")
        sys.exit()
    
    random.seed(args.seed)

    preamble = """
red(X) :- node(X), not green(X), not blue(X).
green(X) :- node(X), not red(X), not blue(X).
blue(X) :- node(X), not red(X), not green(X).

e(X,Y) :- edge(X,Y).
e(Y,X) :- edge(Y,X).

:- e(X,Y), red(X), red(Y).
:- e(X,Y), green(X), green(Y).
:- e(X,Y), blue(X), blue(Y).

qr:- blue(3).

red(1).
green(4).
green(6).
"""

    if args.aspmc:
        preamble = preamble.replace('not', '\+')

    print(preamble)
    
    already_in = [[1, 2], [1, 3], [2, 5], [2, 6], [3, 4], [4, 5], [5, 6]]

    # write the probabilistic facts
    prob_atoms : 'list[float]' = []
    for _ in range(args.n):
        prob = args.prob if (args.prob > 0 and args.prob < 1) else get_random_float()
        prob_atoms.append(prob)
    
    selected_query_atoms = []
    if args.map >= 0:
        n_map = int(args.n * args.map)
        selected_query_atoms = random.sample(range(args.n), n_map)

    # generate the probabilistic facts
    for idx in range(0, args.n - 7):
        n0 = 0
        n1 = n0
        while n0 == n1:
            n0 = random.randint(1, args.n)
            n1 = idx + 7
            l = [n0, n1]
            l.sort()

        # if l in already_in:
        #     n0 = n1

        # while (n1 == n0):
        #     n1 = random.randint(1, args.n)
        #     l = [n0, n1]
        #     l.sort()
        #     if l in already_in:
        #         n1 = n0

        already_in.append(l)

    for idx, l in enumerate(already_in):
        if idx in selected_query_atoms:
            prefix = "map "
        else:
            prefix = ""

        print(f"{prefix}{prob_atoms[idx]}::edge({l[0]},{l[1]}).")

    flat_list = [item for sublist in already_in for item in sublist]
    for j in range(1, max(flat_list) + 1):
        print(f"node({j}).")


if __name__ == "__main__":
    main()