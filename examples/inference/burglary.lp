% from: https://arxiv.org/pdf/2304.00879.pdf

0.5::burglary. 
0.5::earthquake.
alarm :- burglary. 
alarm :- earthquake.
defective :- alarm, not defective.
right :- not alarm.