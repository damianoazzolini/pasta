% From: https://dtai.cs.kuleuven.be/problog/tutorial/learning/04_noisyor.html

#program('
0.0001::word(w1).
0.0001::word(w2).

topic(t1):- p(t1).
topic(t1):- p(t1w1), word(w1).
topic(t1):- p(t1w2), word(w2).

topic(t2):- p(t2).
topic(t2):- p(t2w1), word(w1).
topic(t2):- p(t2w2), word(w2).
').

#learnable(p(t1)).
#learnable(p(t1w1)).
#learnable(p(t1w2)).
#learnable(p(t2)).
#learnable(p(t2w1)).
#learnable(p(t2w2)).

#positive(1,topic(t1)).
#negative(1,topic(t2)).
#positive(1,word(w1)).
#negative(1,word(w2)).

#positive(2,topic(t1)).
#negative(2,topic(t2)).
#positive(2,word(w1)).
#negative(2,word(w2)).

#negative(3,topic(t1)).
#positive(3,topic(t2)).
#negative(3,word(w1)).
#positive(3,word(w2)).

#negative(4,topic(t1)).
#positive(4,topic(t2)).
#positive(4,word(w1)).
#positive(4,word(w2)).

#train(1,2,3,4).
#test(1,2,3,4).

% Expected
% {
% 'word(w1)': 0.0001, 
% 'word(w2)': 0.0001, 
% 'p(t1)': 0.00023166325311871553, 
% 'p(t1w1)': 0.6665507759700737, 
% 'p(t1w2)': 1.8626451492309583e-09, 
% 'p(t2)': 6.926829628368137e-09, 
% 'p(t2w1)': 3.370267898723183e-09, 
% 'p(t2w2)': 0.9999999892402495
% }