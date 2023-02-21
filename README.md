# PASTA: Probabilistic Answer Set programming for STAtistical probabilities

This software allows to perform inference in probabilistic answer set programs under the credal semantics and with statistical statements.
Currentyl supports:
- exact inference
- approximate inference via sampling
- abduction
- map inference
- parameter learning
- lifted inference
- decision theory

## Installation
Requirements: `clingo`, see [install clingo](https://potassco.org/clingo/).
```
python3 -m pip install clingo
```

Then, clone this repository
```
git clone git@github.com:damianoazzolini/pasta.git
```

Finally, move into the `pasta` folder and you are ready to go
```
cd pasta
python3 pasta_solver.py --help
```

See below for some example programs.

You can also use Docker ([image on Dockerhub](https://hub.docker.com/r/damianodamianodamiano/pasta)) to test the application:
```
docker pull damianodamianodamiano/pasta
docker container run -it pasta bash
```
then you are ready to go (follows the nex instructions to run an example).
However, the image is *rarely updated* (almost never), and currently contains an old version (where some bugs are not fixed).

<!-- You can also install the package via `pip`.
Note that there already exists a package called [`pasta`](https://github.com/google/pasta), so this will probably conflict with it if is installed (this happens if you run this in google colab).
```
python3 -m pip install git+https://github.com/damianoazzolini/pasta
``` -->

## How to Use
Use
```
cd pasta
python3 pasta_solver.py --help
```
to see the various available options.

You can also use it as a library
```
import pasta_solver

filename = "../examples/inference/bird_4.lp"
query = "fly(1)"
solver = pasta_solver.Pasta(filename, query)
lp, up = solver.inference()

print("Lower probability for the query " + query + ": " + str(lp))
print("Upper probability for the query " + query + ": " + str(up))
```
where `filename` and `query` are the name of the file and the query to ask.

You can also pass the file as a string, in this way:
```
query = "fly(1)"
solver = pasta_solver.Pasta("", query)  # leave the filename as ""
lp, up = solver.inference(program)

print("Lower probability for the query " + query + ": " + str(lp))
print("Upper probability for the query " + query + ": " + str(up))
```
where `program` is a string containing your program.

You can find more information about the API documentation in the `html/pasta` folder.

### Options
To see the available options, use:
```
python3 pasta_solver.py --help
```

### Exact inference
```
cd pasta
python3 pasta_solver.py ../examples/conditionals/bird_4_cond.lp --query="fly"
```
Asks the query `fly` for the program stored in `../examples/conditionals/bird_4_cond.lp`.
Expected result:
```
Lower probability for the query: 0.7
Upper probability for the query: 1.0
```
You can specify evidence with `--evidence`.

### Abduction
This is still experimental and some features might not work as expected.
```
cd pasta
python3 pasta_solver.py ../examples/abduction/bird_4_abd_prob.lp --query="fly(1)" --abduction
```

### MAP/MPE inference
```
cd pasta
python3 pasta_solver.py ../examples/map/color_map.lp --query="win" --map
```

### Approximate inference
Available: sampling (`--approximate`), gibbs sampling (`--gibbs`), metropolis hastings sampling (`--mh`), rejection sampling (`--rejection`).
```
cd pasta
python3 pasta_solver.py ../examples/inference/bird_4.lp --query="fly(1)" --approximate
```
Use the flag `--samples` to set the number of samples (1000 by default), for example `--samples=2000`.

Use the flag `--processes` to set the number of processes (1 by default), for example `--processes=8`. The maximum number is 16.

## Parameter Learning
```
cd pasta
python3 pasta_solver.py ../examples/learning/background_bayesian_network.lp --pl
```


### Handling Inconsistent Programs
The credal semantics requires that every world has at least one answer set.
These programs are called *consistent*.
Here, we make the same assumption.
<!-- Note that the minimal set of probabilistic facts is correct only if the program satisfies this requirement. -->
<!-- If you don't want to compute this set, use the flag `--no-minimal` or `-nm`. -->
If you ask a query on a program that is not consistent, you should get an error.
You can normalize the probability with the flag `--normalize`.

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
- MAP inference: in press
- Approximate inference: in press
- Lifted inference: under review
- Parameter learning: in press