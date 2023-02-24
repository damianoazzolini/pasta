import sys

'''
Generates programs of the following form with an increasing number
of person. Constant part: stored in the variable 'constant'.
Query: high_number_strokes.
'''

constant = '''
0.4::predisposition_d.
0.6::predisposition_s.

problem(P):- problem_d(P), predisposition_d.
problem(P):- problem_s(P), predisposition_s.

stroke(P) ; not_stroke(P):- problem(P).

:- #count{X:problem(X)} = P, 
   #count{X:stroke(X),problem(X)} = SP,
   10*SP < 4*P.

high_number_strokes:- 
   CS = #count{X : stroke(X)}, CS > 1.
'''

if len(sys.argv) != 2:
    print("Usage: pytohn3 generate_programs_pressure.py <max_size>")
    sys.exit()

prefix = "pressure_inst"


for i in range(2, int(sys.argv[1]) + 1):
    fp = open(f"{prefix}_{i}.lp", "w")
    fp.write(constant)
    
    for ii in range(0, i):
        fp.write(f"d_{ii}:gamma(70.0,1).\n")
        fp.write(f"s_{ii}:gamma(120.0,1).\n")
        fp.write(f"problem_d({ii}):- outside(d_{ii}, 60.0, 80.0).\n")
        fp.write(f"problem_s({ii}):- outside(s_{ii}, 110.0, 130.0).\n")
        fp.write('\n')

    fp.close()
