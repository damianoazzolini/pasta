import paspsp

if __name__ == "__main__":
    filename = "fly_4.lp"
    query = "fly(1)"
    pasp_solver = paspsp.Paspsp(filename, query)
    lp, up = pasp_solver.solve()

    print("Lower probability for the query " + query + ": " + str(lp))
    print("Upper probability for the query " + query + ": " + str(up))
