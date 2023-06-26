% 0.4::b.
% a:gamma(0,1).
% % atom:gaussian(2,3).
% q0 ; q1:- below(a,0.5).
% q0:- below(a,0.7), b.

% q0 ; q1:- below(atom,0.5).
% q0:- between(atom,0.7,1.4), b.

% 0.4::b.
% a:gaussian(0,1).
% q0 ; q1:- below(a,0.5).
% q0:- below(a,0.7), b.

% :- b, below(a,0.2).

% 0.5::ad0.
% 0.6915::f11.
% 0.2139::f21.

% int11 :- f11.
% int21 :- not f11, f21.

% q0 ; q1:- int11.
% q0:- int11, ad0.
% q0:- int21, ad0.

% 0.5::ad2.
% 0.6915::f13.
% 0.2139::f23.

% int13 :- f13.
% int23 :- not f13, f23.

% q0 ; q1:- int13.
% q0:- int13, ad2.
% q0:- int23, ad2.


% 0.5::ad0.
% 0.5::ad2.
% 0.5::ad4.
% 0.5::ad6.
% 0.5::ad8.

% ac1:gaussian(0,1).
% ac3:gaussian(0,1).
% ac5:gaussian(0,1).
% ac7:gaussian(0,1).
% ac9:gaussian(0,1).

% q0 ; q1 :- below(ac1, 0.5).
% q0 ; q1 :- below(ac3, 0.5).
% q0 ; q1 :- below(ac5, 0.5).
% q0 ; q1 :- below(ac7, 0.5).
% q0 ; q1 :- below(ac9, 0.5).

% q0 :- below(ac1, 0.7), ad0.
% q0 :- below(ac3, 0.7), ad2.
% q0 :- below(ac5, 0.7), ad4.
% q0 :- below(ac7, 0.7), ad6.
% q0 :- below(ac9, 0.7), ad8.


% 0.4::b.
% a:gaussian(0,1).
% % a1:gaussian(0,1).

% % % this is the program
% % % ----
% q0:- between(a,0.4,0.5).
% q0:- below(a,0.7), b.
% % % ----
% % q0:- below(a,0.5).
% q0:- below(a,0.4),b.
% q0:- between(a,0.5,0.7),b.
% q0:- between(a,0.4,0.5).




% q0:- below(a,0.5), above(a,0.4).
% q0:- between(a,0.5,0.7), b.
% q0:- below(a,0.4), b.


% qt0:- above(a,0.5).
% q0:- above(a,0.4), nq0.
% nq0:- not qt0.

% q0:- nq1, b.
% nq1:- not qt1.
% qt1:- above(a,0.7).

% 0.4::b.
% a:gaussian(0,3).
% q0;q1:- between(a, 0.0, 1.0).
% q0;q1:-  between(a, 1.5, 2.0).

% 0.4::b.
% a:gaussian(0,1).

% q0:- between(a,0.4,0.5).
% q0:- below(a,1), b.

% q0;q1:- above(a,0.0), below(a,1.0), above(a,1.5).%, below(a,2.0).
% q0:- between(a,0.5,1), b.

% q0:- between(a, 0, -27.0).
% q0:- between(a, -27.0, -28.07).
% q0:- below(a, 0.5).
% q0:- below(a, 0.7), c.
% q0:- below(a, 0.6), b.
% 0.4::b.
% a:gaussian(0,10).
% q0;q1:- between(a, -30, -29.625).
% q0;q1:- between(a, -29.625, 5.608).
% q0;q1:- between(a, 5.608, 25.283).

% q0:- between(a, -30, -29.201).
% q0:- between(a, -29.201, 3.098).
% q0:- between(a, 3.098, 28.362).

% 0.4::b.
% a:gaussian(0,1).
% c:gaussian(0,1).

% q0:- between(a,-4.0, -1.0), between(c,-4.0, -1.0).


% 0.4::b.
% a:gaussian(0,10).
% q0;q1:- between(a, -30, -25.477).
% q0;q1:- between(a, -25.477, -22.356).
% q0;q1:- between(a, -22.356, -21.399).
% q0;q1:- between(a, -21.399, -20.354).
% q0;q1:- between(a, -20.354, -18.628).
% q0;q1:- between(a, -18.628, -15.364).
% q0;q1:- between(a, -15.364, -13.513).
% q0;q1:- between(a, -13.513, -13.156).
% q0;q1:- between(a, -13.156, -11.054).
% q0;q1:- between(a, -11.054, -8.586).
% q0;q1:- between(a, -8.586, -6.211).
% q0;q1:- between(a, -6.211, -4.637).
% q0;q1:- between(a, -4.637, -1.914).
% q0;q1:- between(a, -1.914, -1.468).
% q0;q1:- between(a, -1.468, 1.265).
% q0;q1:- between(a, 1.265, 3.713).
% q0;q1:- between(a, 3.713, 4.303).
% q0;q1:- between(a, 4.303, 6.879).

% q0:- between(a, -30, -24.794).
% q0:- between(a, -24.794, -21.646).
% q0:- between(a, -21.646, -20.621).
% q0:- between(a, -20.621, -18.639).
% q0:- between(a, -18.639, -15.63).
% q0:- between(a, -15.63, -12.887).
% q0:- between(a, -12.887, -11.579).
% q0:- between(a, -11.579, -10.609).
% q0:- between(a, -10.609, -8.859).
% q0:- between(a, -8.859, -5.748).
% q0:- between(a, -5.748, -4.62).
% q0:- between(a, -4.62, -2.379).
% q0:- between(a, -2.379, -1.867).
% q0:- between(a, -1.867, -1.678).
% q0:- between(a, -1.678, 1.183).
% q0:- between(a, 1.183, 3.372).
% q0:- between(a, 3.372, 5.668).
% q0:- between(a, 5.668, 8.168).



% 0.4::predisposition_d.
% 0.6::predisposition_s.

% d_1:gamma(70,1).
% d_2:gamma(70,1).
% d_3:gamma(70,1).
% d_4:gamma(70,1).

% s_1:gamma(120,1).
% s_2:gamma(120,1).
% s_3:gamma(120,1).
% s_4:gamma(120,1).

% problem_d_1:- outside(d_1, 60.0, 80.0).
% problem_d_2:- outside(d_2, 60.0, 80.0).
% problem_d_3:- outside(d_3, 60.0, 80.0).
% problem_d_4:- outside(d_4, 60.0, 80.0).

% problem_s_1:- outside(s_1, 110.0, 130.0).
% problem_s_2:- outside(s_2, 110.0, 130.0).
% problem_s_3:- outside(s_3, 110.0, 130.0).
% problem_s_4:- outside(s_4, 110.0, 130.0).

% problem(1):- problem_d_1, predisposition_d.
% problem(2):- problem_d_2, predisposition_d.
% problem(3):- problem_d_3, predisposition_d.
% problem(4):- problem_d_4, predisposition_d.

% problem(1):- problem_s_1, predisposition_s.
% problem(2):- problem_s_2, predisposition_s.
% problem(3):- problem_s_3, predisposition_s.
% problem(4):- problem_s_4, predisposition_s.

% stroke(P) ; not_stroke(P):- problem(P).

% % :- #count{X:problem(X)} = P, 
% %    #count{X:stroke(X),problem(X)} = SP,
% %    10*SP < 4*P.

% % high_number_strokes:- 
% %    CS = #count{X : stroke(X)}, CS > 1.

% st:- stroke(1).
% st:- stroke(2).
% st:- stroke(3).
% st:- stroke(4).

% % q0:- high_number_strokes.
% q0:- st.

% % Empty with mcintyre loaded
% :- use_module(library(mcintyre)).

% :- mc.

% :- begin_lpad.

% 0.4::predisposition_d.
% 0.6::predisposition_s.

% d_1(X,0):gamma(X,70,1).
% d_2(X,0):gamma(X,70,1).
% d_3(X,0):gamma(X,70,1).
% d_4(X,0):gamma(X,70,1).

% s_1(X,0):gamma(X,120,1).
% s_2(X,0):gamma(X,120,1).
% s_3(X,0):gamma(X,120,1).
% s_4(X,0):gamma(X,120,1).

% problem_d_1:- d_1(X,0), X < 60.0. 
% problem_d_1:- d_1(X,0), X > 80.0.

% problem_d_2:- d_2(X,0), X < 60.0. 
% problem_d_2:- d_2(X,0), X > 80.0.

% problem_d_3:- d_3(X,0), X < 60.0. 
% problem_d_3:- d_3(X,0), X > 80.0.

% problem_d_4:- d_4(X,0), X < 60.0. 
% problem_d_4:- d_4(X,0), X > 80.0.

% problem_s_1:- s_1(X,0), X < 110.0.
% problem_s_1:- s_1(X,0), X > 130.0.

% problem_s_2:- s_2(X,0), X < 110.0.
% problem_s_2:- s_2(X,0), X > 130.0.

% problem_s_3:- s_3(X,0), X < 110.0.
% problem_s_3:- s_3(X,0), X > 130.0.

% problem_s_4:- s_4(X,0), X < 110.0.
% problem_s_4:- s_4(X,0), X > 130.0.

% problem(1):- problem_d_1, predisposition_d.
% problem(2):- problem_d_2, predisposition_d.
% problem(3):- problem_d_3, predisposition_d.
% problem(4):- problem_d_4, predisposition_d.

% problem(1):- problem_s_1, predisposition_s.
% problem(2):- problem_s_2, predisposition_s.
% problem(3):- problem_s_3, predisposition_s.
% problem(4):- problem_s_4, predisposition_s.

% % :- #count{X:problem(X)} = P, 
% %    #count{X:stroke(X),problem(X)} = SP,
% %    10*SP < 4*P.

% % high_number_strokes:- 
% %    CS = #count{X : stroke(X)}, CS > 1.

% stroke(P):- problem(P).


% st:- stroke(1).
% st:- stroke(2).
% st:- stroke(3).
% st:- stroke(4).

% % q0:- high_number_strokes.
% q0:- st.


% :- end_lpad.

% /** <examples> Your example queries go here, e.g.

% ?- mc_sample(query,1000,Prob).

0.4::b.
a:gaussian(0,1).
q0 ; q1:- below(a,0.5).
q0:- below(a,0.7), b.

ne1 :- not e1.
e1:- b, below(a, 0.2).
:- b, below(a, 0.2).
% :- b, between(a, 0.2, 0.3).

% nq0:- not q0.
% 0.4::b.
% 0.6915::f1.
% 0.2139::f2.
% int1 :- f1.
% int2 :- not f1, f2.
% q0 ; q1:- int1.
% q0:- int1, b.
% q0:- int2, b.