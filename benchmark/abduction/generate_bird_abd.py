import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 " + sys.argv[0] + " <lower> <upper>")
        print("For example: python3 " + sys.argv[0] + " 2 20")
        print("will generate 18 (20 - 2) files with name bird_i.lp")
        print("where i ranges from 2 to 20, both included, and with")
        print("the structure of ../../examples/bird_4.lp")
        sys.exit()
    lower = int(sys.argv[1])
    upper = int(sys.argv[2])

    if lower > upper:
        print("Lower must be lower than upper")
        sys.exit()

    det = True
    stress = True
    step = 10 if stress else 1

    for i in range(lower, upper + 1, step):
        filename = "det/bird_" + str(i) + ".lp"
        # filename = "prob/ic_det/bird_" + str(i) + ".lp"
        # filename = "prob/ic_prob/bird_" + str(i) + ".lp"
        f = open(filename, "w")
        for j in range(1, i + 1):
            # f.write("0.5::bird(" + str(j) + ").\n")
            if not det:
                if j % 2 == 0 and j % 3 != 0:
                    f.write("abducible bird(" + str(j) + ").\n")
                elif j % 3 == 0:
                    f.write("0.5::bird(" + str(j) + ").\n")
                else:
                    f.write("bird(" + str(j) + ").\n")
            else:
                # if j % 2 != 0:
                f.write("abducible bird(" + str(j) + ").\n")
                # else:
                # f.write("bird(" + str(j) + ").\n")
        f.write("\nfly(X);nofly(X):- bird(X).\n")
        f.write(
            "\n:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.\n")
        # f.write("\n0.5:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.\n")
        # f.write("\n0.7::fly_w(1).")
        f.write("\nqry:- fly(1).")
        # f.write("\nqry:- fly_w(1).")
        f.close()
