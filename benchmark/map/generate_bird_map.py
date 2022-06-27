base = '''
fly(X);nofly(X):- bird(X).
  
:- #count{X:fly(X),bird(X)} = FB, 
   #count{X:bird(X)} = B, 
   10*FB < 6*B.

'''

init_size = 4
end_size = 30
for s in range(init_size, end_size + 1):
    fp = open(f"bird_{s}_map.lp", "w")
    fp.write(base)
    for j in range(1, s + 1):
        if j % 2 == 0:
            fp.write(f"0.5::bird({j}).\n")
        else:
            fp.write(f"map 0.5::bird({j}).\n")
    fp.close()

    fp = open(f"bird_{s}_mpe.lp", "w")
    fp.write(base)
    for j in range(1, s + 1):
        fp.write(f"map 0.5::bird({j}).\n")
    fp.close()


