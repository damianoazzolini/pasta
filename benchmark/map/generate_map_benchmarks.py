import argparse
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
        "program",
        help="Type of programs to generate.",
        type=str,
        choices=["1","2", "3", "4"]
    )

    command_parser.add_argument(
        "-n",
        help="Number of probabilistic facts",
        type=int,
        required=True
    )

    command_parser.add_argument(
        "--mpe",
        help="For MPE (all query atoms)",
        action="store_true"
    )

    command_parser.add_argument(
        "--seed",
        help="Seed for random numbers generator",
        type=int,
        default=42
    )

    return command_parser.parse_args()

def get_random_float() -> float:
    # truncate at 3 decimals
    return float(str(random.random())[:5])


def generate_first_type_programs(args : argparse.Namespace):
    """
    # generate rules, where the negation is random
    qr0 ; nqr0 :- not a0.
    qr0 :- not a1.
    qr2 ; nqr2 :-  a2.
    qr2 :-  a3.
    ... 
    """
    for i in range(args.n):
        if args.mpe:
            # print(f"map {get_random_float()}::a{i}.")
            print(f"{get_random_float()}::a{i}.")
    
    for i in range(0,args.n):
        prefix = "" if random.random() > 0.5 else "not"
        if i % 2 == 0:
            print(f"qr{i} ; nqr{i} :- {prefix} a{i}.")
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
    for i in range(args.n):
        if args.mpe:
            print(f"map {get_random_float()}::a{i}.")
    
    prefix_pf = "" if random.random() > 0.5 else "not"
    print(f"qr0 ; nqr0:- {prefix_pf} a0.")
    for i in range(1,args.n):
        prefix = "" if random.random() > 0.5 else "n"
        prefix_pf = "" if random.random() > 0.5 else "not"
        if i % 2 == 0:
            print(f"qr{i} ; nqr{i} :- qr{i-1}, {prefix_pf} a{i}.")
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
    for i in range(args.n):
        if args.mpe:
            print(f"map {get_random_float()}::a{i}.")
    
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
        hh = ";".join(h)
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
            prefix = "" if random.random() > 0.5 else "not"
            at = f"{prefix} a{i}"
            if random.random() > 0.5:
                b.append(at)
            
        return ','.join(b)
        
    
    for i in range(args.n):
        if args.mpe:
            print(f"map {get_random_float()}::a{i}.")
    
    bb = generate_body()
    print(f"qr ; nqr :- {bb}.")
    bb = generate_body()
    print(f"qr :- {bb}.")
    

def main():
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