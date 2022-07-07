import math
import random

base = '''
valuable(X) ; not_valuable(X):- gold(X).
:- #count{X:valuable(X), gold(X)} = FB, #count{X:gold(X)} = B, 10*FB < 6*B.

:- not valuable(1).
:- not gold(1).

wp(P):- PS = #sum{X,Y : gold(Y,X)}, PNS = #sum{X,Y : not_gold(Y,X)}, P = PS + PNS.

#maximize{P : wp(P)}.

#show gold/2.
#show not_gold/2.

'''


init_size = 4
end_size = 30
for s in range(init_size, end_size + 1):
    fp = open(f"gold_{s}_clingo_mpe.lp", "w")
    fp.write(base)
    for j in range(1, s + 1):
        p = random.random()
        # p = 0.5
        prob = str((math.log(p))*1000)[:5].replace('.', '')
        n_prob = str((math.log(1 - p))*1000)[:5].replace('.', '')
        fp.write("0{gold(" + str(j) + ")}1.\n")
        fp.write(f"gold({j},{prob}):- gold({j}).\n")
        fp.write(f"not_gold({j},{n_prob}):- not gold({j}).\n")
    fp.close()
