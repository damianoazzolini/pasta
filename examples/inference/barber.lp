worksInTown(X) :- barber(X).
worksInRiver(X) :- fsherman(X).
worksInTown(X) :- person(X), not worksInRiver(X).
worksInRiver(X) :- person(X), not worksInTown(X).
shaves(X, Y) :- barber(X), person(Y), not shaves(Y, Y ).

person(j).

0.5::barber(j).
0.5::fsherman(j).


qra:- barber(j).
nqra:- not barber(j).

qrb:- fisherman(j).
nqrb:- not fisherman(j).

qrc :- worksInTown(j).
nqrc :- not worksInTown(j).

qrd :- worksInRiver(j).
nqrd :- not worksInRiver(j).

qre:- shaves(j,j).
nqre:- not shaves(j,j).