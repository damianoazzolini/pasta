% 0.5::shops(a).
% 0.5::shops(b).
% 0.5::shops(c).
% 0.5::shops(d).

% % price_spaghetti:gaussian(700,10).
% % price_steak:gaussian(800,10).
% % salary(a):gaussian(1800,10).
% % salary(b):gaussian(1500,10).

% % bought_spaghetti(a):- #constraint(price_spaghetti < salary(a)).
% % bought_steak(a):- #constraint(price_steak < salary(a)).
% % bought_both(a):- bought_spaghetti(a), bought_steak(a).
% % % :- bought_both(a), #constraint(price_spaghetti + price_steak > salary(a)).

% % bought_spaghetti(b):- #constraint(price_spaghetti < salary(b)).
% % bought_steak(b):- #constraint(price_steak < salary(b)).
% % bought_both(b):- bought_spaghetti(b), bought_steak(b).
% % % :- bought_both(b), #constraint(price_spaghetti + price_steak > salary(b)).

% % qr:- #count{X : bought_both(X)} > 0.
% qr:- #count{X : bought(spaghetti,X)} > 0.
% % qr:- bought_spaghetti(a).

% salary(a):gaussian(1200,10).
% good_salary(a):- #constraint(salary(a) > 1220).
% average_salary(a):- #constraint(salary(a) > 1200).

% salary(b):gaussian(1200,10).
% good_salary(b):- #constraint(salary(b) > 1220).
% average_salary(b):- #constraint(salary(b) > 1200).

% salary(c):gaussian(1200,10).
% good_salary(c):- #constraint(salary(c) > 1220).
% average_salary(c):- #constraint(salary(c) > 1200).

% salary(d):gaussian(1200,10).
% good_salary(d):- #constraint(salary(d) > 1220).
% average_salary(d):- #constraint(salary(d) > 1200).

% bought(spaghetti,a) ; bought(steak,a) :- shops(a), good_salary(a).
% bought(onion,a) ; bought(bread,a) :- shops(a), average_salary(a).

% bought(spaghetti,b) ; bought(steak,b) :- shops(b), good_salary(b).
% bought(onion,b) ; bought(bread,b) :- shops(b), average_salary(b).

% bought(spaghetti,c) ; bought(steak,c) :- shops(c), good_salary(c).
% bought(onion,c) ; bought(bread,c) :- shops(c), average_salary(c).

% bought(spaghetti,d) ; bought(steak,d) :- shops(d), good_salary(d).
% bought(onion,d) ; bought(bread,d) :- shops(d), average_salary(d).


% cs(C):- #count{X:bought(spaghetti,X)} = C0, #count{X : bought(onion,X)} = C1, C = C0 + C1.
% ce(C):- #count{X,Y : bought(Y,X)} = C.

% :- cs(S), ce(C), 10* S < 4*C.
