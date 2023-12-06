# 0.03::a0.
# 0.03::a1.
# 0.03::a2.
# 0.03::a3.
# 0.03::a4.
# 0.03::a5.
# 0.03::a6.
# 0.03::a7.
# 0.03::a8.
# 0.03::a9.

# qr:- a0.
# qr :- a1, \+ nqr.
# nqr :- a1, \+ qr.
# qr:- a2.
# qr :- a3, \+ nqr.
# nqr :- a3, \+ qr.
# qr:- a4.
# qr :- a5, \+ nqr.
# nqr :- a5, \+ qr.
# qr:- a6.
# qr :- a7, \+ nqr.
# nqr :- a7, \+ qr.
# qr:- a8.
# qr :- a9, \+ nqr.
# nqr :- a9, \+ qr.


# generates the programs
import argparse

command_parser = argparse.ArgumentParser()
command_parser.add_argument("max_size", help="max size", type=int, default=15)
command_parser.add_argument("--negated", help="negated version", action="store_true", default=False)

args = command_parser.parse_args()


for sz in range(10, args.max_size + 1, 5):
    for n_opt in range(2, sz + 1):
        if args.negated:
            fp = open(f"qr_nqr_clauses_negated_{sz}_{n_opt}.lp", "w")
        else:
            fp = open(f"qr_nqr_clauses_{sz}_{n_opt}.lp", "w")
        
        for j in range(0, n_opt):
            if args.negated:
                if j % 2 == 0:
                    fp.write(f"optimizable [0.85,0.95]::a{j}.\n")
                else:
                    fp.write(f"optimizable [0.05,0.15]::a{j}.\n")
            else:
                fp.write(f"optimizable [0.05,0.25]::a{j}.\n")
        for j in range(n_opt, sz):
            if args.negated:
                if j % 2 == 0:
                    fp.write(f"0.97::a{j}.\n")
                else:
                    fp.write(f"0.03::a{j}.\n")
            else:
                fp.write(f"0.03::a{j}.\n")
        
        fp.write("\n")
        
        for j in range(0, sz):
            if j % 2 == 0:
                if args.negated:
                    fp.write(f"qr:- not a{j}.\n")
                else:
                    fp.write(f"qr:- a{j}.\n")
            else:
                fp.write(f"qr:- a{j}, not nqr.\n")
                fp.write(f"nqr:- a{j}, not qr.\n")
                
        
        fp.close()

# generates the runners
fnames_list : 'list[str]' = []

for sz in range(10, args.max_size + 1, 5):
    for alg in ["COBYLA","SLSQP"]:
        for eps in [0,0.05]:
            s = "no_eps" if eps == 0 else "eps"
            if args.negated:
                fname = f"runner_clauses_negated_{sz}_{alg}_{s}.sh"
            else:
                fname = f"runner_clauses_{sz}_{alg}_{s}.sh"
            fp = open(fname, "w")
            fnames_list.append(fname)
            
            fp.write(
f'''
#!/bin/bash
#SBATCH --job-name=qr_c{sz}
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --partition=longrun
#SBATCH --output=qr_c{sz}_{alg}_{s}.log

echo "Started at: "
date

for i in `seq 2 {sz - 1}`; do 
echo "instance $i"
time pasta qr_nqr_clauses_{"negated_" if args.negated else ""}{sz}_$i.lp --query="qr" --optimize {"--epsilon=0.05" if eps == 0.5 else ""} --threshold=0.7 --target=upper --verbose --method={alg}
done 

echo "Ended at: "
date
''')
        
            fp.close()


# generates the sbatcher
if args.negated:
    fname = f"sbatcher_clauses_negated.sh"
else:
    fname = f"sbatcher_clauses.sh"

fp = open(fname, "w")

for l in fnames_list:
    fp.write("sbatch " + l)
    fp.write("\n")

fp.close()
