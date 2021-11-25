dom_bird_a(1..2).
bird_a(1,693):-v_bird_a_(1).
not_bird_a(1,693):- not v_bird_a_(1).
bird_a(2,693):-v_bird_a_(2).
not_bird_a(2,693):- not v_bird_a_(2).
bird_a(I) :- bird_a(I,_).
not_bird_a(I) :- not_bird_a(I,_).
#show bird_a/2.
#show not_bird_a/2.
0{v_bird_a_(I):dom_bird_a(I)}2.

dom_bird_b(1..2).
bird_b(1,916):-v_bird_b_(1).
not_bird_b(1,510):- not v_bird_b_(1).
bird_b(2,916):-v_bird_b_(2).
not_bird_b(2,510):- not v_bird_b_(2).
bird_b(I) :- bird_b(I,_).
not_bird_b(I) :- not_bird_b(I,_).
#show bird_b/2.
#show not_bird_b/2.
0{v_bird_b_(I):dom_bird_b(I)}2.

fly_a(X);nofly_a(X):-bird_a(X).
fly_b(X);nofly_b(X):-bird_b(X).

:-#count{X:fly_a(X),bird_a(X)}=FBA,#count{X:fly_b(X),bird_b(X)}=FBB,FB=FBA+FBB,#count{X:bird_a(X)}=BA,#count{X:bird_b(X)}=BB,B=BA+BB,10*FB<6*B.

fly_1:-fly_a(1).
fly_1:-fly_b(1).

q:- fly_1.
#show q/0.

nq:- not fly_1.
#show nq/0.