stress = True

upper = 1000 if stress is True else 50
step = 10 if stress is True else 1


for i in range(10, upper, step):
    f_det = open(f'det/dummy_{i}.lp', 'w')

    if stress is False:
        f_prob_det = open(f'prob_det/dummy_{i}.lp', 'w')
        f_prob_prob = open(f'prob_prob/dummy_{i}.lp', 'w')

    for j in range(1, i+1):
        f_det.write(f"abducible a({j}).\n")

        if stress is False:
            if j % 2 == 0:
                f_prob_det.write(f"abducible a({j}).\n")  # type: ignore
                f_prob_prob.write(f"abducible a({j}).\n")  # type: ignore
            else:
                f_prob_det.write(f"0.5::a({j}).\n")  # type: ignore
                f_prob_prob.write(f"0.5::a({j}).\n")  # type: ignore

        f_det.write(f"qr:- a({j}).\n")

        if stress is False:
            f_prob_det.write(f"qr:- a({j}).\n")  # type: ignore
            f_prob_prob.write(f"qr:- a({j}).\n")  # type: ignore

    f_det.write("qrr:- qr.\n")

    if stress is False:
        f_prob_det.write("qrr:- qr.\n")  # type: ignore
        f_prob_prob.write("qrr:- qr.\n")  # type: ignore

    f_det.write("c(C):- #count{X : a(X)} = C.\n:- c(C), C < 2.\n")

    if stress is False:
        f_prob_det.write("c(C):- #count{X : a(X)} = C.\n:- c(C), C < 2.\n")  # type: ignore
        f_prob_prob.write("c(C):- #count{X : a(X)} = C.\n0.9:- c(C), C < 2.\n")  # type: ignore
