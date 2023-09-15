# PASTA: Probabilistic Answer Set programming for STAtistical probabilities

This software allows to perform inference in probabilistic answer set programs under the credal semantics (PASPs) and with statistical statements.
Currently it supports:
- exact inference
- approximate inference via sampling
- abduction
- map inference
- parameter learning
- lifted inference
- decision theory

## Installation
```
git clone https://github.com/damianoazzolini/pasta
cd pasta
pip install .
```
or
```
pip install git+https://github.com/damianoazzolini/pasta
```

You can also use Docker ([image on Dockerhub](https://hub.docker.com/r/damianodamianodamiano/pasta)) to test the application:
```
docker container run -it damianodamianodamiano/pasta bash
```
then you are ready to go (follows the next instructions to run an example).


## How to Use
After the installation, use
```
pasta --help
```
to see the available options.

### Exact inference
The probability of a query in a PASP is given by a range $P(q) = [\underline{P}(q),\overline{P}(q)]$ where $\underline{P}(q) = \sum_{w_i \mid |AS(w_i)| > 0 \ \land \ m \in AS(w_i), \ m \models q} P(w_i)$
and
$\overline{P}(q) = \sum_{w_i \mid \exists m \in AS(w_i), \ m \models q} P(w_i)$
where $P(w)$ is the probability of the world $w$ computed as $P(w) = \prod_{i \mid f_i = \top} \Pi_i \cdot \prod_{i \mid f_i = \bot} (1 - \Pi_i)$
and $AS(w)$ is the set of answer sets for a world $w$.
The current algorithm adopts projected answer set enumeration to solve the task.

Example:
```
pasta examples/conditionals/bird_4_cond.lp --query="fly"
```
Asks the query `fly` for the program stored in `examples/conditionals/bird_4_cond.lp`.
Expected result:
```
Lower probability for the query: 0.7
Upper probability for the query: 1.0
```
You can specify evidence with `--evidence`.

### Abduction
This is still experimental and some features might not work as expected.
```
pasta examples/abduction/bird_4_abd_prob.lp --query="fly(1)" --abduction
```

### MAP/MPE inference
```
pasta examples/map/color_map.lp --query="win" --map
```

### Approximate inference
Available: sampling (`--approximate`), gibbs sampling (`--gibbs`), metropolis hastings sampling (`--mh`), rejection sampling (`--rejection`).
```
pasta examples/inference/bird_4.lp --query="fly(1)" --approximate
```
Use the flag `--samples` to set the number of samples (1000 by default), for example `--samples=2000`.

Use the flag `--processes` to set the number of processes (1 by default), for example `--processes=8`. The maximum number is 16.

### Parameter Learning
```
pasta examples/learning/background_bayesian_network.lp --pl
```

### Decision Theory
```
pasta examples/decision_theory/dummy.lp -dt
```
For normalization, you should use `-dtn` insted of `-dt`.

### Handling Inconsistent Programs
The credal semantics requires that every world has at least one answer set.
These programs are called *consistent*.
Here, we make the same assumption.

If you ask a query on a program that is not consistent, you should get an error.
You can normalize the probability with the flag `--normalize`.

### Use PASTA as a Library
You can also use it as a library
```
from pasta.pasta_solver import Pasta

filename = "examples/inference/bird_4.lp"
query = "fly(1)"
solver = Pasta(filename, query)
lp, up = solver.inference()

print("Lower probability for the query " + query + ": " + str(lp))
print("Upper probability for the query " + query + ": " + str(up))
```
where `filename` and `query` are the name of the file and the query to ask.

You can also pass the file as a string, in this way:
```
program = """
0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
0.5::bird(4).

% A bird can fly or not fly
0{fly(X)}1 :- bird(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.
"""

query = "fly(1)"

solver = Pasta("", query)  # leave the filename as ""
lp, up = solver.inference(program)

print(f"Lower probability: {lp}")
print(f"Upper probability: {up}")
```
where `program` is a string containing your program.

All the above tasks can be used with the Python interface.

### Caveat
Make sure to not write clauses with the same functor of probabilistic facts.
For example, you should not write:
```
0.4::c(1).
c(X):- a(X).
```
In other words, probabilistic facts cannot appear as head atoms of any rule.


## Syntax
Basically, PASTA (PASP) programs are ASP programs plus probabilistic facts.
Probabilistic facts can be added with the syntax: `prob::atom.` where `prob` is a floating point number (0 < number <= 1) and `atom` is a standard ASP fact.
For example, `0.5::a.` states that `a` has probability `0.5`.

For more examples, see the `examples` folder.

Using exact inference, you can also express statistical statements (x% of the y elements share the same behavior) with the syntax: `(A | B)[LP,UP].`
For example, "at least 60% of the birds fly" can be expressed with
```
(fly(X) | bird(X))[0.6,1].
```
See `examples/conditionals/bird_4_cond.lp` for an example.

Note: be super careful when using rules with disjunction in the head.
You should replace them with choice rules.

## Description and How to Cite
The system and the various types of inferences are currently described in:
- Exact inference and statistical staments: `Damiano Azzolini, Elena Bellodi, and Fabrizio Riguzzi. Statistical statements in probabilistic logic programming. In Georg Gottlob, Daniela Inclezan, and Marco Maratea, editors, Logic Programming and Nonmonotonic Reasoning, pages 43--55, Cham, 2022. Springer International Publishing.`
- Abduction (preliminary): `Damiano Azzolini, Elena Bellodi, and Fabrizio Riguzzi. Abduction in (probabilistic) answer set programming. In Roberta Calegari, Giovanni Ciatto, and Andrea Omicini, editors, Proceedings of the 36th Italian Conference on Computational Logic, volume 3204 of CEUR Workshop Proceedings, pages 90--103, Aachen, Germany, 2022. Sun SITE Central Europe.`
- MAP inference: `Damiano Azzolini, Elena Bellodi, and Fabrizio Riguzzi. Map inference in probabilistic answer set programs. In Agostino Dovier, Angelo Montanari, and Andrea Orlandini, editors, AIxIA 2022 -- Advances in Artificial Intelligence, pages 413--426, Cham, 2023. Springer International Publishing.`
- Approximate inference: `Damiano Azzolini, Elena Bellodi, and Fabrizio Riguzzi. Approximate inference in probabilistic answer set programming for statistical probabilities. In Agostino Dovier, Angelo Montanari, and Andrea Orlandini, editors, AIxIA 2022 -- Advances in Artificial Intelligence, pages 33--46, Cham, 2023. Springer International Publishing.` 
- Lifted inference: under review
- Parameter learning: in press
- Decision theory (preliminary): in press