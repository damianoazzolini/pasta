0.4::sunshine.
0.8::warm.
0.3::tired.
0.6::wind.

run :- not walk, sunshine, not wind.
walk :- not run.
sick :- run, tired.
sick :- not warm.

query(sick).

% Expected result
% Lower probability for the query: 0.2
% Upper probability for the query: 0.2384