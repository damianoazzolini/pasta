from pastasolver.pasta_solver import Pasta

filename = ""

program = """
0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
0.5::bird(4).

% A bird can fly or not fly
0{fly(X)}1 :- bird(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.
"""

solver = Pasta(filename=filename,query="fly(1)")

lp, up = solver.inference(program)

print(f"Lower probability: {lp}")
print(f"Upper probability: {up}")