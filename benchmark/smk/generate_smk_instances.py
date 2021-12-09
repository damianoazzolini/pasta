import sys
import networkx
import random

# see examples/conditionals/smokers.lp for the structure

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 " + sys.argv[0] + " <lower> <upper>")
        print("For example: python3 " + sys.argv[0] + " 2 20")
        print("will generate 18 (20 - 2) files with name smk_i.lp")
        print("where i ranges from 2 to 20, both included, and with")
        print("the structure of ../../examples/smokers.lp")
        sys.exit()
    lower = int(sys.argv[1])
    upper = int(sys.argv[2])

    if lower > upper:
        print("Lower must be lower than upper")
        sys.exit()

    # i instance (number of probabilistic facts) number
    # j dataser number, since results are averages of 10 runs

    # graph_size = 10
    for i in range(lower, upper + 1):
        for j in range(0,10):
            filename = "smk_" + str(i) + "/smk_" + str(j) + ".lp"
            f = open(filename, "w")
            ba = networkx.barabasi_albert_graph(i, 3)
            le = []
            for a in ba.edges:
                f.write("0.5::friend(" + str(a[0]) + "," + str(a[1]) + ").\n")
                le.append(a[0])
                le.append(a[1])

            f.write("\n:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.\n")

            f.write("\nsmokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).\n\n")
            # selects 50% of the smokers
            smokers = random.sample(list(set(le)),int(i/2))

            ls = []
            for el in smokers:
                ls.append(el)
                f.write("smokes(" + str(el) + ").\n")

            lns = list(filter(lambda x: x not in ls,list(set(le))))

            f.write("\nquery(smokes(" + str(random.sample(lns,1)[0]) + ")).")
            
            # f.write("\nfly:- fly(_).")
            f.close()
