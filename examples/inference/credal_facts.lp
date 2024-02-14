% From https://kamel.ime.usp.br/pages/learn_dpasp#installation

work(X) :- not sleep(X), person(X).
sleep(X) :- not work(X), not stress(X), person(X).
optimizable [0.0, 0.3]::stress(anna).
optimizable [0.2, 0.7]::stress(bill).
optimizable [0.5, 0.9]::stress(charlie).
0.2::lead(anna); 0.45::lead(bill); 0.35::lead(charlie).
calls(X, Y) :- not sleep(X), not sleep(Y), lead(X), person(Y), X != Y.
person(anna).
person(bill).
person(charlie).
% #query work(anna).
% #query sleep(bill).
% #query calls(anna, bill).
% #semantics credal.

% qr:- work(anna).
qr:- sleep(bill).
% qr:- calls(anna, bill).

% pasta credal_facts.lp --query="qr" -cf --verbose --method=SLSQP --verbose
