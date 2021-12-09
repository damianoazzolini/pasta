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

    for i in range(lower,upper + 1):
        filename = "bird_" + str(i) + ".lp"
        f = open(filename,"w")
        for j in range(1, i + 1):
            f.write("0.5::bird(" + str(j) + ").\n")
        f.write("\nfly(X);nofly(X):- bird(X).\n")
        f.write("\n:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.\n")
        f.write("\n0.7::fly_w(1).")
        f.write("\nfly:- fly(1).")
        f.write("\nfly:- fly_w(1).")
        f.close()
