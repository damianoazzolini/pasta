'''
The smokers dataset from SMProbLog.
'''

# import argparse

# command_parser = argparse.ArgumentParser()
# command_parser.add_argument("--sm", help="SMProbLog version", action="store_true")

# args = command_parser.parse_args()

base = '''
0.1 :: asthma_f(1).
0.1 :: asthma_f(2).

0.3 :: stress(1).
0.3 :: stress(2).

0.4 :: stress_fact(1).
0.4 :: stress_fact(2).

smokes(X):- stress(X), stress_fact(X).

smokes(X) :- influences(Y, X), smokes(Y ).


0.4 :: asthma_fact(1).
0.4 :: asthma_fact(2).

asthma_rule(X):- smokes(X), asthma_fact(X).

asthma(X):- asthma_f(X).
asthma(X):- asthma_rule(X).

:- smokes(X), asthma(X).

0.3 :: influences(1, 2). 
0.6 :: influences(2, 1).

qr:- smokes(1).
'''


fp = open("smokers_0.lp","w", encoding="utf-8")
fp.write(base)
fp.write("\nquery(smokes(1)).\n")
fp.close()

fp = open("smokers_1.lp","w", encoding="utf-8")
fp.write(base)
s = '''
0.1 :: asthma_f(3).
0.3 :: stress(3).
0.4 :: stress_fact(3).
0.4 :: asthma_fact(3).
'''
fp.write("\nquery(smokes(1)).\n")
fp.close()

fp = open("smokers_2.lp","w", encoding="utf-8")
fp.write(base)
s += '''
0.1 :: asthma_f(4).
0.3 :: stress(4).
0.4 :: stress_fact(4).
0.4 :: asthma_fact(4).
'''
fp.write("\nquery(smokes(1)).\n")
fp.close()

fp = open("smokers_3.lp", "w", encoding="utf-8")
fp.write(base)
s += '''
0.2::influences(2,3).
'''
fp.write("\nquery(smokes(1)).\n")
fp.close()

fp = open("smokers_4.lp", "w", encoding="utf-8")
fp.write(base)
s += '''
0.7::influences(3,4).
'''
fp.write("\nquery(smokes(1)).\n")
fp.close()

fp = open("smokers_5.lp", "w", encoding="utf-8")
fp.write(base)
s += '''
0.9::influences(4,1).
'''
fp.write("\nquery(smokes(1)).\n")
fp.close()