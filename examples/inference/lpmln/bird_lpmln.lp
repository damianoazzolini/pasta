bird(X) :- residentbird(X).
bird(X) :- migratorybird(X).
:- residentbird(X), migratorybird(X).
2::residentbird(jo).
1::migratorybird(jo).

% python3 pasta_solver.py temp.pl  --all --lpmln
% : 0.09003057317038046
% migratorybird(jo) bird(jo): 0.24472847105479764
% residentbird(jo) bird(jo): 0.6652409557748219