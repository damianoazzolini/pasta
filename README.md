# PASTA: Probabilistic Answer Set programming for STAtistical probabilities

## Syntax
Probabilistic facts can be added with the syntax: `prob::atom.` where `prob` is a floating point number (0 < number <= 1) and `atom` is a standard ASP fact.
For example, `0.5::a.` states that `a` has probability `0.5`.
It is possible to use intervals to define probabilistic fact.
For example, writing `0.5::f(1). 0.5::f(2). 0.5::f(3).` is equivalent to writing `0.5::f(1..3).`.
However, the latter (intervals) are preferred since give better performance (due to the program conversion).

*Note: q/0 and nq/0 are internally used, so they cannot be declared as (probabilistic) facts (atoms) in the program*.

For more examples, see the `examples` folder.