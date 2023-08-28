import argparse


EXAMPLE_STRING_EXACT = "pasta \
    examples/inference/bird_4.lp \
    --query=\"fly(1)\""
EXAMPLE_STRING_EXACT_EVIDENCE = "pasta \
    examples/inference/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\""
EXAMPLE_STRING_APPROXIMATE = "pasta \
    examples/inference/bird_4.lp \
    --query=\"fly(1)\" \
    --approximate"
EXAMPLE_STRING_APPROXIMATE_EVIDENCE = "pasta \
    examples/inference/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\" --rejection"
EXAMPLE_STRINGS = "Examples:\n\n" + EXAMPLE_STRING_EXACT + "\n\n" + \
    EXAMPLE_STRING_EXACT_EVIDENCE + \
    "\n\n" + EXAMPLE_STRING_APPROXIMATE + "\n\n" + EXAMPLE_STRING_APPROXIMATE_EVIDENCE

PASTA_DESCRIPTION = \
    "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"


def parse_args_wrapper():
    '''
    Wrapper for arguments parsing.
    '''
    command_parser = argparse.ArgumentParser(
        description=PASTA_DESCRIPTION,
        epilog=EXAMPLE_STRINGS
    )

    command_parser.add_argument(
        "filename",
        help="Program to analyse",
        type=str
    )
    command_parser.add_argument(
        "-q",
        "--query",
        help="Query",
        type=str, default=""
    )
    command_parser.add_argument(
        "-e",
        "--evidence",
        help="Evidence",
        type=str,
        default=""
    )
    command_parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose mode, default: false",
        action="store_true"
    )
    command_parser.add_argument(
        "--pedantic",
        help="Pedantic mode (prints the converted program and all the worlds),\
            default: false",
        action="store_true"
    )

    # approximate
    command_parser.add_argument(
        "--approximate",
        help="Compute approximate probability",
        action="store_true"
    )
    command_parser.add_argument(
        "--samples",
        help="Number of samples, default 1000",
        type=int,
        default=1000
    )
    command_parser.add_argument("--processes",
        help="Number of processes",
        type=int,
        default=1
    )
    command_parser.add_argument(
        "--mh",
        help="Use Metropolis Hastings sampling",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--gibbs",
        help="Use Gibbs Sampling sampling",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--block",
        help="Set the block value for Gibbs sampling",
        type=int,
        default=1
    )
    command_parser.add_argument(
        "--rejection",
        help="Use rejection Sampling sampling",
        action="store_true",
        default=False
    )

    command_parser.add_argument(
        "--pl",
        help="Parameter learning",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--abduction",
        help="Abduction",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--map",
        help="MAP (MPE) inference",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--upper",
        help="Select upper probability as target",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--minimal",
        "-nm",
        help="Compute the minimal set of probabilistic facts",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--normalize",
        help="Normalize the probability if some worlds have no answer sets",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--stop-if-inconsistent",
        "-sif",
        help="Raise an error if some worlds have no answer sets (and lists them)",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    command_parser.add_argument(
        "--solver",
        help="Uses an ASP solver for the task",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--one",
        help="Compute only 1 solution for MAP. Currently has no effects",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--xor",
        help="Uses XOR constraints for approximate inference",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--alpha",
        help="Constant for approximate inferece with XOR constraints. Default = 0.004",
        type=float,
        default=0.004
    )
    command_parser.add_argument(
        "--delta",
        help="Accuracy for approximate inferece with XOR constraints. Default = 2",
        type=float,
        default=2
    )

    # decision theory
    command_parser.add_argument(
        "-dtn",
        help="Decision theory (naive)",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "-dtopt",
        help="Decision theory with optimization",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "-dt",
        help="Decision theory (improved)",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--no-mix",
        help="Compute the utility of a strategy by considering\
            only the lower probability and upper probability for\
            the lower and upper utility bounds respectively.",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--lpmln",
        help="Use the lpmnl semantics",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--all",
        help="Computes the weights for all the answer sets",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--test",
        help="Check the consistency by sampling: 1 stops when an\
            inconsistent world is found, 0 keeps sampling.",
        type = int,
        choices=range(0,2)
    )
    command_parser.add_argument(
        "--uxor",
        help="Check the consistency by XOR sampling.",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--profile",
        help="Use code profiling (cProfile)",
        action="store_true",
        default=False
    )

    # for decision theory approximate with genetic algorithm
    command_parser.add_argument(
        "--popsize",
        help="Population size (default 50).",
        type=int,
        default=50
    )
    command_parser.add_argument(
        "--mutation",
        help="Mutation probability (default 0.05).",
        type=float,
        default=0.05
    )
    command_parser.add_argument(
        "--iterations",
        help="Iterations for the genetic algorithm (default 1000)",
        type=int,
        default=1000
    )

    # optimizable/reducible task
    command_parser.add_argument(
        "--optimize",
        help="Optimize the probability of optimizable facts.",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--reducible",
        help="Reduce the probabilistic answer set program.",
        action="store_true",
        default=False
    )
    command_parser.add_argument(
        "--target",
        help="Target for the optimizable/reducible task (default upper).",
        default="upper",
        choices=["upper","lower"]
    )
    command_parser.add_argument(
        "--threshold",
        help="Threshold \\tau for the optimizable task (q > \\tau, default -1).",
        default=-1
    )
    command_parser.add_argument(
        "--epsilon",
        help="Bound for pairwise constraints (default -1).",
        type=float,
        default=-1
    )
    command_parser.add_argument(
        "--method",
        help="Optimization algorithm to use (default SLSQP).",
        default="SLSQP",
        choices=["SLSQP","COBYLA"]
    )
    command_parser.add_argument(
        "--chunk",
        help="Split the symbolic equation in chunks with this size (default 100).",
        type = int,
        default=100
    )


    return command_parser.parse_args()
