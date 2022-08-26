# PASTA: Probabilistic Answer Set programming for STAtistical probabilities

This software allows to perform inference in probabilistic answer set programs under the credal semantics and with statistical statements.

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

You can also use Docker ([image on Dockerhub](https://hub.docker.com/repository/docker/damianodamianodamiano/pasta)) to test the application:
```
docker pull damianodamianodamiano/pasta
docker container run -it pasta bash
```
then you are ready to go (follows the nex instructions to run an example).
However, the image is not always updated.

<!-- You can also install the package via `pip`.
Note that there already exists a package called [`pasta`](https://github.com/google/pasta), so this will probably conflict with it if is installed (this happens if you run this in google colab).
```
python3 -m pip install git+https://github.com/damianoazzolini/pasta
``` -->

## Usage
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
```
python3 pasta_solver.py --help
usage: pasta_solver.py [-h] [-q QUERY] [-e EVIDENCE] [-v] [--pedantic]
                       [--approximate] [--samples SAMPLES] [--mh] [--gibbs]
                       [--block BLOCK] [--rejection] [--pl] [--abduction]
                       [--map] [--upper] [--no-minimal] [--normalize]
                       [--stop-if-inconsistent]
                       filename

PASTA: Probabilistic Answer Set programming for STAtistical probabilities

positional arguments:
  filename              Program to analyse

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY Query
  -e EVIDENCE, --evidence EVIDENCE Evidence
  -v, --verbose         Verbose mode, default: false
  --pedantic            Pedantic mode, default: false
  --approximate         Compute approximate probability
  --samples SAMPLES     Number of samples, default 1000
  --mh                  Use Metropolis Hastings sampling
  --gibbs               Use Gibbs Sampling sampling
  --block BLOCK         Set the block value for Gibbs sampling
  --rejection           Use rejection Sampling sampling
  --pl                  Parameter learning
  --abduction           Abduction
  --map                 MAP (MPE) inference
  --upper               Select upper probability for MAP and abduction
  --no-minimal          Do not compute the minimal set of probabilistic facts
  --normalize           Normalize the probability if some worlds do not have answer sets
  --stop-if-inconsistent Raise an error if a world without answer sets is found
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
The papers describing this system will be soon available.