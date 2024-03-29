import sys
import random

'''
0.4::b.
a:gaussian(0,30).
q0 ; q1:- below(a,0.5).
q0:- below(a,0.7), b.

Every instance of size $n$ is characterized by 1 discrete 
probabilistic fact $b$ and 1 continuos probabilistic 
fact $a$, $n/2$ rules \texttt{q0 ; q1:- between(a, lb_i, ub_i).}, 
with 1 between(a, lb_i, ub_i) for i in [1,n/2], and $n/2$ rules 
\texttt{q0:- between(a, lb_i, ub_i), b.}, with 1 between(a, lb_i, ub_i) 
for i in [1,n/2].
lb_i < ub_i forall i and ub_i < lb_{i+1}.

Query: q0.
'''

if len(sys.argv) != 2:
    print("Usage: python3 generate_programs_t2.py <max_size>")
    sys.exit()

prefix = "t2_inst"

for i in range(1, int(sys.argv[1])):
    c1 : 'str' = ""
    c2 : 'str' = ""
    fc1 : 'list[str]' = []
    fc2 : 'list[str]' = []
    prev0 = -30
    prev1 = random.uniform(-30, -20)
    
    for ii in range(0, i):
        if ii == 0:
            current = prev1
        else:
            current = random.uniform(prev0, prev0 + float(60/i))
        fc1.append(f"between(a, {round(prev0,3)}, {round(current,3)})")    
        prev0 = current
    
    prev0 = -30
    prev1 = random.uniform(-30, -20)
    for ii in range(0, i):
        if ii == 0:
            current = prev1
        else:
            current = random.uniform(prev0, prev0 + float(60/i))
        fc2.append(f"between(a, {round(prev0,3)}, {round(current,3)})")    
        prev0 = current

    fp = open(f"{prefix}_{i}.lp", "w")

    fp.write("0.4::b.\na:gaussian(0,10).\n")

    for el in fc1:
        fp.write(f"q0;q1:- {el}.\n")
    fp.write('\n')
    for el in fc2:
        fp.write(f"q0:- {el}.\n")
    fp.write('\n')

    fp.close()