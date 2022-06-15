# PASTA: Probabilistic Answer Set programming for STAtistical probabilities

This software allows to perform inference in probabilistic answer set programs under the credal semantics and with statistical statements.

## Usage

Exact inference
```
cd src/pasta
python3 pasta_solver.py ./../examples/conditionals/bird_4_cond.lp --query="fly"
```

Abduction
```
cd src/pasta
python3 pasta_solver.py ./../examples/abduction/bird_4_abd_prob.lp --query="fly(1)" --abduction
```

## Syntax
Probabilistic facts can be added with the syntax: `prob::atom.` where `prob` is a floating point number (0 < number <= 1) and `atom` is a standard ASP fact.
For example, `0.5::a.` states that `a` has probability `0.5`.

For more examples, see the `examples` folder.

## Description and How to Cite
The papers describing this system will be soon available.