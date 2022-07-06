import random

base = '''
valuable(X) ; not_valuable(X):- gold(X).
:- #count{X:valuable(X), gold(X)} = FB, #count{X:gold(X)} = B, 10*FB < 6*B.

'''

init_size = 4
end_size = 30
for s in range(init_size, end_size + 1):
    fp = open(f"gold_{s}_map.lp", "w")
    fp.write(base)
    for j in range(1, s + 1):
        prob = str(random.random())[:5]
        if j % 2 == 0:
            fp.write(f"{prob}::gold({j}).\n")
        else:
            fp.write(f"map {prob}::gold({j}).\n")
    fp.close()

    fp = open(f"gold_{s}_mpe.lp", "w")
    fp.write(base)
    for j in range(1, s + 1):
        prob = str(random.random())[:5]
        fp.write(f"map {prob}::gold({j}).\n")
    fp.close()


