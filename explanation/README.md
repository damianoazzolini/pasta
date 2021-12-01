# Step by step
Here, we show two examples, one with a query and one with query and evidence.

## Query without evidence
The program `../examples/bird_2_2.lp` with query `fly_1` is first converted into the one saved in `bird_step_0.lp`.
This is sent into `clingo`, and the following command is executed:
```
clingo bird_step_0.lp -e cautious
```
The cautious consequences are empty, so nothing will be further inserted into `bird_step_1.lp`.
`bird_step_1.lp` is the second program used to compute the lower and upper probability.
The command is:
```
clingo bird_step_1.lp 0 --project
```
The output is (already grouped and with the associated probability):
```
bird_a(1) -> a_1 -> 0.5
bird_a(2) -> a_2 -> 0.5
bird_b(1) -> b_1 -> 0.4
bird_b(2) -> b_2 -> 0.4

% 4 birds -> 1 world
a_1 a_2 b_1 b_2: prob: 0.5*0.5*0.4*0.4 = 0.04 -> l + u

% 3 birds -> 4 worlds
a_2 b_1 b_2: prob: 0.5*0.5*0.4*0.4 = 0.04 -> u
a_1 b_1 b_2: prob: 0.5*0.5*0.4*0.4 = 0.04 -> l + u
a_1 a_2 b_1: prob: 0.5*0.5*0.6*0.4 = 0.06 -> l + u
a_1 a_2 b_2: prob: 0.5*0.5*0.4*0.6 = 0.06 -> u

% 2 birds -> 6 worlds
a_1 a_2: prob: 0.5*0.5*0.6*0.6 = 0.09 -> l + u
a_1 b_1: prob: 0.5*0.5*0.4*0.6 = 0.06 -> l + u
a_1 b_2: prob: 0.5*0.5*0.6*0.4 = 0.06 -> l + u
a_2 b_1: prob: 0.5*0.5*0.4*0.6 = 0.06 -> l + u
a_2 b_2: prob: 0.5*0.5*0.6*0.4 = 0.06
b_1 b_2: prob: 0.5*0.5*0.4*0.4 = 0.04 -> l + u

% 1 birds -> 4 worlds
a_1: prob: 0.5*0.5*0.6*0.6 = 0.09 -> l + u
a_2: prob: 0.5*0.5*0.6*0.6 = 0.09
b_1: prob: 0.5*0.5*0.4*0.6 = 0.06 -> l + u
b_2: prob: 0.5*0.5*0.6*0.4 = 0.06

% 0 birds -> 1 world
```
The lower probability is: `0.04*3 + 0.06*5 + 0.09*2 = 0.6`.

The upper probability is: `0.04*4 + 0.06*6 + 0.09*2 = 0.7`.

## Query with evidence
Consider the program `../examples/bird_4.lp` with query `fly(1)` and evidence `fly(2)`.
Suppose the probabilities of the facts are:
```
0.11::bird(1).
0.13::bird(2).
0.17::bird(3).
0.19::bird(4).
```
In this way, all the probabilities multiplied by 100 are prime numbers.
Instead of the query, we now need to consider the evidence.
The evidence could be true or false, but it must be true at least in one model of every world.
As before, we look for the minimal subset of facts that make the query true.
We found that `bird(2)` must be true.
We generate all the possible projected models from the modified program with added the constraint `:- not bird(2).` with the same operations as before.
We get
```

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

uq / (uq + le) = 0.11325576343136133
lq / (lq + ue) = 0.07395299999999999
```
The last two are the upper and lower probabilities for the query given evidence.
