import clingo


# def setup_clingo_delta(mode: int, lb: int, ub: int, consequent: str, fixed_a : str = "", fixed_b : str = "") -> 'clingo.Control':
#     '''
#     mode = 0 lower - cautious
#     mode = 1 upper - brave
#     '''
#     if mode == 0:
#         ctl = clingo.Control(["--enum-mode=cautious"])
#     else:
#         ctl = clingo.Control(["--enum-mode=brave"])

#     ctl.add('base', [], consequent)

#     return ctl

def delta_ax(mode: int, n_prob_facts: int, lb: int, ub: int, query: str):
    '''
    mode = 0 lower - cautious
    mode = 1 upper - brave
    '''
    if mode == 0:
        ctl = clingo.Control(["--enum-mode=cautious"])
    else:
        ctl = clingo.Control(["--enum-mode=brave"])

    ctl.add('base', [], "a(1).")
    ctl.add('base', [], "0{ c(X)  }1 :-  a(X).")
    
    if lb != 0:
        ctl.add('base', [
        ], ":- #count{X: a(X)} = H, #count{X:c(X) , a(X)} = FH, 100*FH < " + str(lb) + "*H.")

    if ub != 100:
        ctl.add('base', [
        ], ":- #count{X: a(X)} = H, #count{X:c(X) , a(X)} = FH, 100*FH > " + str(ub) + "*H.")

    for i in range(2, n_prob_facts + 1):
        ctl.add('base', [], f"a({i}).")
    ctl.ground([("base", [])])

    opt_m: str = ""
    with ctl.solve(yield_=True) as handle:  # type: ignore
        for m in handle:  # type: ignore
            opt_m = str(m)  # type: ignore
            handle.get()  # type: ignore

    return 1 if query in opt_m.split(' ') else 0


def delta_cx_axbxy(mode: int, lw: 'str', lb: int, ub: int, query: str):
    '''
    mode = 0 lower - cautious
    mode = 1 upper - brave
    1 if the query is true in at least one answer set for a world with 
        $i$ probabilistic facts true and $k$ overlaps, 0 otherwise.
        This world is stored in lw.
    '''
    if mode == 0:
        ctl = clingo.Control(["--enum-mode=cautious"])
    else:
        ctl = clingo.Control(["--enum-mode=brave"])

    ctl.add('base', [], "a(1).")
    ctl.add('base', [], "b(1,1).")
    ctl.add('base', [], "0{ c(X) }1 :-  a(X), b(X,Y).")

    if lb != 0:
        ctl.add('base', [], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X),a(X),b(X,Y)} = FH, 100*FH < " + str(lb) + "*H.")
    if ub != 100:
        ctl.add('base', [], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X),a(X),b(X,Y)} = FH, 100*FH > " + str(ub) + "*H.")

    for i in range(0, int(len(lw) / 2)):
        if int(lw[i]) == 1:
            ctl.add('base', [], f"a({i + 2}).")
        if int(lw[i + int(len(lw) / 2)]) == 1:
            ctl.add('base', [], f"b({i + 2}, 1).")
    ctl.ground([("base", [])])

    opt_m: str = ""
    with ctl.solve(yield_=True) as handle:  # type: ignore
        for m in handle:  # type: ignore
            opt_m = str(m)  # type: ignore
            handle.get()  # type: ignore

    return 1 if query in opt_m.split(' ') else 0


def delta_cx_axbxy_k(mode: int, lw: 'tuple[int]', lb: int, ub: int, query: str):
    '''
    mode = 0 lower - cautious
    mode = 1 upper - brave
    1 if the query is true in at least one answer set for a world with 
        ki1 random variables with index 1 set to true, ki2 random variables 
        with index 2 set to true, and so on.
        The world is stored in lw.
    '''

    if int(lw[0]) * int(lw[int(len(lw)/2)]) == 0:
        # since the query is c(1), if at least one a(1) and one b(1,_)
        # is 0, the probability is 0
        # print('zero')
        return 0

    if mode == 0:
        ctl = clingo.Control(["--enum-mode=cautious"])
    else:
        ctl = clingo.Control(["--enum-mode=brave"])
    
    ctl.add('base', [], "0{ c(X) }1 :-  a(X), b(X,Y).")

    if lb != 0:
        ctl.add('base', [
        ], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X),a(X),b(X,Y)} = FH, 100*FH < " + str(lb) + "*H.")

    if ub != 100:
        ctl.add('base', [
        ], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X),a(X),b(X,Y)} = FH, 100*FH > " + str(ub) + "*H.")

    for i in range(0, int(len(lw))):
        for k in range(0, int(lw[i])):
            if i < int(len(lw) / 2):
                ctl.add('base', [], f"a({i + 1}).")
            else:
                ctl.add('base', [], f"b({i + 1 - int(len(lw) / 2)}, {k}).")
    ctl.ground([("base", [])])

    opt_m: str = ""
    with ctl.solve(yield_=True) as handle:  # type: ignore
        for m in handle:  # type: ignore
            opt_m = str(m)  # type: ignore
            handle.get()  # type: ignore

    return 1 if query in opt_m.split(' ') else 0


def delta_cxy_axbxy_k(mode: int, lw: 'tuple[int]', lb: int, ub: int, query: str = "c(1,1)"):
    '''
    mode = 0 lower - cautious
    mode = 1 upper - brave
    1 if the query is true in at least one answer set for a world with 
        ki1 random variables with index 1 set to true, ki2 random variables 
        with index 2 set to true, and so on.
        The world is stored in lw.
    '''
    
    # TODO: uguale a delta_cx_axbxy_k, cambia solo c(X) -> c(X,Y)

    if int(lw[0]) * int(lw[int(len(lw)/2)]) == 0:
        # since the query is c(1), if at least one a(1) and one b(1,_)
        # is 0, the probability is 0
        # print('zero')
        return 0

    if mode == 0:
        ctl = clingo.Control(["--enum-mode=cautious"])
    else:
        ctl = clingo.Control(["--enum-mode=brave"])
    
    ctl.add('base', [], "0{ c(X,Y) }1 :-  a(X), b(X,Y).")

    if lb != 0:
        ctl.add('base', [
        ], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X,Y),a(X),b(X,Y)} = FH, 100*FH < " + str(lb) + "*H.")

    if ub != 100:
        ctl.add('base', [
        ], ":- #count{X,Y: a(X),b(X,Y)} = H, #count{X,Y:c(X,Y),a(X),b(X,Y)} = FH, 100*FH > " + str(ub) + "*H.")

    for i in range(0, int(len(lw))):
        for k in range(1, int(lw[i]) + 1):
            if i < int(len(lw) / 2):
                ctl.add('base', [], f"a({i + 1}).")
            else:
                ctl.add('base', [], f"b({i + 1 - int(len(lw) / 2)}, {k}).")
    ctl.ground([("base", [])])

    opt_m: str = ""
    with ctl.solve(yield_=True) as handle:  # type: ignore
        for m in handle:  # type: ignore
            opt_m = str(m)  # type: ignore
            handle.get()  # type: ignore

    return 1 if query in opt_m.split(' ') else 0