# Step by step
The program `bird.lp` with query `fly` is first converted into the one saved in `bird_step_0.lp`.
This is sent into `clingo`, and the following command is executed:
```
clingo bird_step_0.lp -e cautious
```
The cautious consequences are empty, so nothing is further inserted into `bird_step_1.lp`.
`bird_step_1.lp` is the second program used to compute the lower and upper probability.
The command is:
```
clingo bird_step_1.lp 0 --project
```
The output is (already grouped and with the associated probability):
```
% 1 1 2 0 -> 3218 -> 0.04
bird_a(2) bird_b(1) bird_b(2) nobird_a(1) model_not_query(1,693,1,693,2,1832,0,0)
bird_a(2) bird_b(1) bird_b(2) nobird_a(1) model_query(1,693,1,693,2,1832,0,0)

% 1 1 1 1 -> 2812 -> 0.06
bird_a(2) bird_b(1) nobird_a(1) nobird_b(2) model_query(1,693,1,693,1,916,1,510)
bird_a(2) bird_b(2) nobird_a(1) nobird_b(1) model_not_query(1,693,1,693,1,916,1,510)
bird_a(1) bird_b(2) nobird_a(2) nobird_b(1) model_query(1,693,1,693,1,916,1,510)
bird_a(1) bird_b(1) nobird_a(2) nobird_b(2) model_query(1,693,1,693,1,916,1,510)

% 1 1 0 2 -> 2406 -> 0.09
bird_a(2) nobird_a(1) nobird_b(1) nobird_b(2) model_not_query(1,693,1,693,0,0,2,1020)
bird_a(1) nobird_a(2) nobird_b(1) nobird_b(2) model_query(1,693,1,693,0,0,2,1020)

% 0 2 2 0 -> 3218 -> 0.04 -- LOWER
bird_b(1) bird_b(2) nobird_a(1) nobird_a(2) model_query(0,0,2,1386,2,1832,0,0)

% 0 2 1 1 -> 2812 -> 0.06
bird_b(2) nobird_a(1) nobird_a(2) nobird_b(1) model_not_query(0,0,2,1386,1,916,1,510)
bird_b(1) nobird_a(1) nobird_a(2) nobird_b(2) model_query(0,0,2,1386,1,916,1,510)

nobird_a(1) nobird_a(2) nobird_b(1) nobird_b(2) model_not_query(0,0,2,1386,0,0,2,1020) % 0 2 0 2 -> 2406 -> 0.09 -- NOT QUERY

% 2 0 2 0 -> 3218 -> 0.04 -- LOWER
bird_a(1) bird_a(2) bird_b(1) bird_b(2) model_query(2,1386,0,0,2,1832,0,0)

% 2 0 1 1 -> 2812 -> 0.06
bird_a(1) bird_a(2) bird_b(2) nobird_b(1) model_query(2,1386,0,0,1,916,1,510)
bird_a(1) bird_a(2) bird_b(2) nobird_b(1) model_not_query(2,1386,0,0,1,916,1,510)
bird_a(1) bird_a(2) bird_b(1) nobird_b(2) model_query(2,1386,0,0,1,916,1,510)

% 2 0 0 2 -> 2406 -> 0.09 -- LOWER
bird_a(1) bird_a(2) nobird_b(1) nobird_b(2) model_query(2,1386,0,0,0,0,2,1020)

% 1 1 2 0 -> 3218 -> 0.04 -- LOWER
bird_a(1) bird_b(1) bird_b(2) nobird_a(2) model_query(1,693,1,693,2,1832,0,0)
```
The lower probability is: `0.04 + 0.04 + 0.09 + 0.04 = 0.21`.

The upper probability is: `0.04 + 0.06*3 + 0.09 + 0.04 + 0.06 + 0.04 + 0.06*2 + 0.09 + 0.04 = 0.70`.