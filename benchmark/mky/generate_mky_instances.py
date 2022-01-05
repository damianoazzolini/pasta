import sys
import networkx
import random
import time

# see examples/conditionals/mky.lp for the structure

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 " + sys.argv[0] + " <lower> <upper>")
        print("For example: python3 " + sys.argv[0] + " 2 20")
        print("will generate 18 (20 - 2) files with name mky_i.lp")
        print("where i ranges from 2 to 20, both included, and with")
        print("the structure of ../../examples/mky.lp")
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
        for j in range(0, 3):
            filename = "mky_" + str(i) + "/mky_" + str(j) + ".lp"
            f = open(filename, "w")
            ba = networkx.barabasi_albert_graph(i, 3)
            le = []
            for a in ba.edges:
                f.write("0.5::r(" + str(a[0]) + "," + str(a[1]) + ").\n")
                le.append(a[0])
                le.append(a[1])

            f.write("\nf1(Y) ; not_f1(Y):- h(Y).\n")

            f.write(":- #count{X:h(X)} = H,#count{X:f1(X),h(X)} = FH, 10*FH < 2*H.\n")
            
            f.write("\nf2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).\n")

            f.write(":- #count{X:h(X),r(X,_)} = H, #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH, 10*FH < 9*H.\n")

            f.write("qry:- f1(A).\nqry:- f2(A, B).\n\n")

            # selects 50% of the elements
            elements = random.sample(list(set(le)), int(i/2))

            ls = []
            for el in elements:
                ls.append(el)
                f.write("0.5::h(" + str(el) + ").\n")

            lns = list(filter(lambda x: x not in ls, list(set(le))))

            f.write("\nquery(qry).")

            # f.write("\nfly:- fly(_).")
            f.close()
