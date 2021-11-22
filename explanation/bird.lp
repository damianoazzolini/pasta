bird_a(1..4).
bird_b(1..4).

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

% query fly
