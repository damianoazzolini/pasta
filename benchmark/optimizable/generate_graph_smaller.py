l_5 = ["edge(1,2)", "edge(1,3)", "edge(2,4)", "edge(3,5)", "edge(4,6)"]

l_6 = l_5 + ["edge(5,6)"]

l_7 = l_6 + ["edge(6,7)"]

l_8 = l_7 + ["edge(6,8)"]

l_9 = l_8 + ["edge(7,9)"]

l_10 = ["edge(1,2)", "edge(1,3)", "edge(2,4)", "edge(3,5)", "edge(4,6)", "edge(5,6)", "edge(6,7)", "edge(6,8)", "edge(7,9)", "edge(8,9)"]

l_11 = l_10 + ["edge(9,10)"]

l_12 = l_11 + ["edge(9,11)"]

l_13 = l_12 + ["edge(10,12)"]

l_14 = l_13 + ["edge(11,13)"]

l_probs = [0.5,
0.5,
0.8121440014439971,
0.9318293737581665,
0.9615464115892465,
0.9624207154646245,
0.8785836646800782,
0.994983574039669,
0.840237422823531,
0.877123958103951,
0.8281664113721792,
0.8945639242705665,
0.9448434461234041,
0.9104238155813139,
0.944787245341966]

l_sz = [l_5,l_6,l_7,l_8,l_9,l_10,l_11,l_12,l_13,l_14]
# l_fnames = [filename_5, filename_6, filename_7, filename_8, filename_9, filename_10, filename_11, filename_12, filename_13, filename_14]
# filename_25 = "inst_25.lp"

dest = [6,6,7,8,9,9,10,11,12,13,13]

baseline = '''

path(X,X):- node(X).
path(X,Y):- path(X,Z), edge(Z,Y).

transmit(A,B):- path(A,B), node(A), node(B), not not_transmit(A,B).
not_transmit(A,B):- path(A,B), node(A), node(B), not transmit(A,B).

'''

for l_facts in l_sz:
    # un file per ogni n \in 2..n_facts
    for n_opt in range(2, len(l_facts) + 1):
        fp = open(f"graph_{len(l_facts)}_{n_opt}.lp", "w")
        
        for j in range(0, n_opt):
            fp.write(f"optimizable [0.4,0.95]::{l_facts[j]}.\n")
        for j in range(n_opt, len(l_facts)):
            fp.write(f"{l_probs[j]}::{l_facts[j]}.\n")
        
    
        fp.write(baseline)
        
        n_nodes = dest[len(l_facts) - 5]
        
        fp.write(f"qr:- transmit(1,{n_nodes}).\n")
        
        for j in range(1, n_nodes + 1):
            fp.write(f"node({j}).\n")


# generates the runners
fnames_list : 'list[str]' = []

for sz in range(5, 15):
    for alg in ["COBYLA","SLSQP"]:
        for eps in [0,0.05]:
            s = "no_eps" if eps == 0 else "eps"
            fname = f"runner_graph_{sz}_{alg}_{s}.sh"
            fp = open(fname, "w")
            fnames_list.append(fname)
            
            fp.write(
f'''
#!/bin/bash
#SBATCH --job-name=graph{sz}
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --partition=longrun
#SBATCH --output=graph_{sz}_{alg}_{s}.log

echo "Started at: "
date

for i in `seq 2 {sz}`; do 
echo "instance $i"
time pasta graph_{sz}_$i.lp --query="qr" --optimize {"--epsilon=0.05" if eps == 0.5 else ""} --threshold=0.7 --target=upper --verbose --method={alg}
done 

echo "Ended at: "
date
''')
        
            fp.close()


# generates the sbatcher
fp = open("sbatcher_graph.sh", "w")

for l in fnames_list:
    fp.write("sbatch " + l)
    fp.write("\n")

fp.close()
