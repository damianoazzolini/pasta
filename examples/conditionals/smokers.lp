% Two people (indexed with a, ..., e) know each other with a certain probability
0.11::friend(a,b).
0.13::friend(b,c).
0.17::friend(c,e).
0.19::friend(b,d).
0.23::friend(d,e).

% Induced graph
%           c
%         /   \ 
%    a - b      e
%         \   /
%           d

(smokes(Y) | smokes(X), friend(X,Y))[0.4,1].

% % The conditional is exploded into
% 0{smokes(Y)}1:- smokes(X), friend(X,Y).

% :- #count{Y,X:smokes(X),friend(X,Y)} = F,
%    #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF,
%    10*SF < 4*F.

% However, we know that d smokes with probability 0.7
0.7::smokes_w(d).

% We know that some people smoke
smokes(a).
smokes(c).
smokes(e).

smk:- smokes_w(d).
smk:- smokes(d).

% Example query: smk
% Expected: lower: 0.7, upper: 0.706269