# 0.95::a0.
# 0.95::a1.
# 0.95::a2.
# 0.95::a3.
# 0.95::a4.
# 0.95::a5.
# 0.95::a6.
# 0.95::a7.
# 0.95::a8.
# 0.95::a9.

# qr:- a0, a2, a4, a6, a8.
# qr:- a1, a3, a5, a7, a9, \+ nqr.
# nqr:- a1, a3, a5, a7, a9, \+ qr.


# generates the programs
import argparse

command_parser = argparse.ArgumentParser()
command_parser.add_argument("max_size", help="max size", type=int, default=15)
command_parser.add_argument("--negated", help="negated version", action="store_true", default=False)

args = command_parser.parse_args()


for sz in range(10, args.max_size + 1, 5):
    for n_opt in range(2, sz + 1):
        if args.negated:
            fp = open(f"qr_nqr_rules_negated_{sz}_{n_opt}.lp", "w")
        else:
            fp = open(f"qr_nqr_rules_{sz}_{n_opt}.lp", "w")
        
        for j in range(0, n_opt):
            if args.negated:
                if j % 2 == 0:
                    fp.write(f"optimizable [0.01,0.11]::a{j}.\n")
                else:
                    fp.write(f"optimizable [0.89,0.99]::a{j}.\n")
            else:
                fp.write(f"optimizable [0.4,0.99]::a{j}.\n")
        for j in range(n_opt, sz):
            if args.negated:
                if j % 2 == 0:
                    fp.write(f"0.01::a{j}.\n")
                else:
                    fp.write(f"0.99::a{j}.\n")
            else:
                fp.write(f"0.99::a{j}.\n")
        
        fp.write("\n")
        
        body_qr : 'list[str]' = []
        body_qr_nqr : 'list[str]' = []
        for j in range(0, sz):
            if j % 2 == 0:
                if args.negated:
                    body_qr.append(f"not a{j}")
                else:
                    body_qr.append(f"a{j}")
            else:
                body_qr_nqr.append(f"a{j}")
            
        qr_clause = "qr:- " + ', '.join(body_qr) + ".\n"
        qr_nqr_clause = "qr:- " + ', '.join(body_qr_nqr) + ", not nqr.\n"
        nqr_qr_clause = "nqr:- " + ', '.join(body_qr_nqr) + ", not qr.\n"
        
        fp.write(qr_clause)
        fp.write(qr_nqr_clause)
        fp.write(nqr_qr_clause)
        
        fp.close()

# generates the runners
fnames_list : 'list[str]' = []

for sz in range(10, args.max_size + 1, 5):
    for alg in ["COBYLA","SLSQP"]:
        for eps in [0,0.05]:
            s = "no_eps" if eps == 0 else "eps"
            if args.negated:
                fname = f"runner_rules_negated_{sz}_{alg}_{s}.sh"
            else:
                fname = f"runner_rules_{sz}_{alg}_{s}.sh"
            fp = open(fname, "w")
            fnames_list.append(fname)
            
            fp.write(
f'''
#!/bin/bash
#SBATCH --job-name=qr_r{sz}
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --partition=longrun
#SBATCH --output=qr_r{sz}_{alg}_{s}.log

echo "Started at: "
date

for i in `seq 2 {sz - 1}`; do 
echo "instance $i"
time pasta qr_nqr_rules_{"negated_" if args.negated else ""}{sz}_$i.lp --query="qr" --optimize {"--epsilon=0.05" if eps == 0.5 else ""} --threshold=0.7 --target=upper --verbose --method={alg}
done 

echo "Ended at: "
date
''')
        
            fp.close()


# generates the sbatcher
if args.negated:
    fname = f"sbatcher_rules_negated.sh"
else:
    fname = f"sbatcher_rules.sh"

fp = open(fname, "w")

for l in fnames_list:
    fp.write("sbatch " + l)
    fp.write("\n")

fp.close()
