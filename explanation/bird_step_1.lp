dom_a(1..2).
0{v_a(I):dom_a(I)}2.

dom_b(1..2).
0{v_b(I):dom_b(I)}2.

bird_a(1,693):- v_a(1).
nobird_a(1,693):- not v_a(1).

bird_a(2,693):- v_a(2).
nobird_a(2,693):- not v_a(2).

% bird_a(3,693):- v_a(3).
% nobird_a(3,693):- not v_a(3).

% bird_a(4,693):- v_a(4).
% nobird_a(4,693):- not v_a(4).


bird_b(1,916):- v_b(1).
nobird_b(1,510):- not v_b(1).

bird_b(2,916):- v_b(2).
nobird_b(2,510):- not v_b(2).

% bird_b(3,916):- v_b(3).
% nobird_b(3,510):- not v_b(3).

% bird_b(4,916):- v_b(4).
% nobird_b(4,510):- not v_b(4).

bird_a(X):- bird_a(X,_).
nobird_a(X):- nobird_a(X,_).

bird_b(X):- bird_b(X,_).
nobird_b(X):- nobird_b(X,_).

fly_a(X);nofly_a(X):- bird_a(X).
fly_b(X);nofly_b(X):- bird_b(X).

:-  #count{X:fly_a(X),bird_a(X)} = FBA,
    #count{X:fly_b(X),bird_b(X)} = FBB,
    FB = FBA + FBB,
    #count{X:bird_a(X)} = BA, 
    #count{X:bird_b(X)} = BB, 
    B = BA + BB,
    10*FB<6*B.

fly:- fly_a(1).
fly:- fly_b(1).

model_query(BA,SBA,NBA,SNBA,BB,SBB,NBB,SNBB) :-
    BA = #count { X :bird_a(X) },
    NBA = #count { X: nobird_a(X) },
    SBA = #sum { Y,X : dom_a(X), bird_a(X,Y) },
    SNBA = #sum { Y,X : dom_a(X), nobird_a(X,Y) },

    BB = #count { X :bird_b(X) },
    NBB = #count { X: nobird_b(X) },
    SBB = #sum { Y,X : dom_b(X), bird_b(X,Y) },
    SNBB = #sum { Y,X : dom_b(X), nobird_b(X,Y) },
    fly.

model_not_query(BA,SBA,NBA,SNBA,BB,SBB,NBB,SNBB) :-
    BA = #count { X :bird_a(X) },
    NBA = #count { X: nobird_a(X) },
    SBA = #sum { Y,X : dom_a(X), bird_a(X,Y) },
    SNBA = #sum { Y,X : dom_a(X), nobird_a(X,Y) },

    BB = #count { X :bird_b(X) },
    NBB = #count { X: nobird_b(X) },
    SBB = #sum { Y,X : dom_b(X), bird_b(X,Y) },
    SNBB = #sum { Y,X : dom_b(X), nobird_b(X,Y) },
    not fly.

#show bird_a/1.
#show nobird_a/1.
#show bird_b/1.
#show nobird_b/1.
#show model_query/8.
#show model_not_query/8.