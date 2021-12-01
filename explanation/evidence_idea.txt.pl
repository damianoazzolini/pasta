Questo va bene
Bisogna trovare le variabili che devono essere vere per l'evidenza,
poi aggiungere

evvvvv:- fly(2).
#show evvvvv/0.
ne :- not fly(2).
#show ne/0.

e poi controllare i vari modelli

bird(2,130) bird(1,110) bird(3,170) bird(4,190) ne q
bird(2,130) bird(1,110) bird(3,170) bird(4,190) evvvvv q
bird(2,130) bird(1,110) bird(3,170) bird(4,190) evvvvv nq
uq + ue -> 0.00046189


bird(2,130) bird(1,110) bird(3,170) ne q not_bird(4,810)
bird(2,130) bird(1,110) bird(3,170) evvvvv nq not_bird(4,810)
bird(2,130) bird(1,110) bird(3,170) evvvvv q not_bird(4,810)
uq + ue -> 0.00196911


bird(2,130) bird(3,170) bird(4,190) ne nq not_bird(1,890)
bird(2,130) bird(3,170) bird(4,190) evvvvv nq not_bird(1,890)
ue -> 0.003737110


bird(2,130) bird(1,110) bird(4,190) ne q not_bird(3,830)
bird(2,130) bird(1,110) bird(4,190) evvvvv nq not_bird(3,830)
bird(2,130) bird(1,110) bird(4,190) evvvvv q not_bird(3,830)
ue uq -> 0.00225511


bird(2,130) bird(1,110) evvvvv q not_bird(4,810) not_bird(3,830)
uq lq -> 0.00961389


bird(2,130) bird(3,170) evvvvv nq not_bird(4,810) not_bird(1,890)
ue le -> 0.01593189

bird(2,130) bird(4,190) evvvvv nq not_bird(3,830) not_bird(1,890)
ue le -> 0.01824589

bird(2,130) evvvvv nq not_bird(4,810) not_bird(3,830) not_bird(1,890)
ue le -> 0.07778511


uq = 0.00046189 + 0.00196911 + 0.00225511 + 0.00961389 = 0.0143
lq = 0.00961389
ue = 0.00046189 + 0.00196911 + 0.003737110 + 0.00225511 +  0.01593189 +  0.01824589 + 0.07778511 = 0.12038611
le = 0.01593189 + 0.01824589 + 0.07778511 = 0.11196289