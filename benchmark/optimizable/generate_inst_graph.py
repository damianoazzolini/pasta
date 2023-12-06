import random
import copy

e_10 = [
    "edge(1,2).",
    "edge(1,3).",
    "edge(2,4).",
    "edge(3,5).",
    "edge(4,6).",
    "edge(5,6).",
    "edge(6,7).",
    "edge(6,8).",
    "edge(7,9).",
    "edge(8,9)."
]
e_15 = copy.deepcopy(e_10)

e_15.extend([
    "edge(9,10).",
    "edge(9,11).",
    "edge(10,12).",
    "edge(11,13).",
    "edge(12,13)."
])

e_20 = copy.deepcopy(e_15)
e_20.extend([
    "edge(13,14).",
    "edge(14,17).",
    "edge(13,15).",
    "edge(15,16).",
    "edge(17,16)."
])

e_25 = copy.deepcopy(e_20)
e_25.extend([
    "edge(17,18).",
    "edge(17,19).",
    "edge(18,20).",
    "edge(19,21).",
    "edge(20,21)."
])


for inst, sz, last_node in zip([e_10,e_15,e_20,e_25], [10,15,20,25], [9,13,17,21]):
    for sub_inst in range(1, sz + 1):
        fp = open(f"{sz}/{sz}_{sub_inst}.lp", "w")
        
        v = sub_inst
        for e in inst:
            if v > 0:
                fp.write(f"optimizable [0.8,0.95]::{e}\n")
                v -= 1
            else:
                r = random.random()
                while r < 0.8:
                    r = random.random()
                fp.write(f"{r}::{e}\n")
        
        fp.write(
''' 

path(X,X):- node(X).
path(X,Y):- path(X,Z), edge(Z,Y).

{transmit(A,B)}:- path(A,B), node(A), node(B).

:- #count{A,B:transmit(A,B),path(A,B)} = RB, #count{A,B:path(A,B)} = R, 100*RB < 30*R.

'''
        )
        
        fp.write(f"qr:- transmit(1,{last_node}).\n")
        
        for node in range(1, last_node + 1):
            fp.write(f"node({node}).\n")
        
        fp.close()
                