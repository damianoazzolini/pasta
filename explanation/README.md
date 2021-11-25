# Step by step
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