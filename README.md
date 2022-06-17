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

Finally, move into the src/pasta folder and you are ready to go
```
cd src/pasta
python3 pasta_solver.py --help
```

See below for some example programs.

You can also use Docker ([image on Dockerhub](https://hub.docker.com/repository/docker/damianodamianodamiano/pasta)) to test the application:
```
docker pull damianodamianodamiano/pasta
docker container run -it pasta bash
```
then you are ready to go (follows the nex instructions to run an example).

You can also install the package via `pip`.
Note that there already exists a package called [`pasta`](https://github.com/google/pasta), so this will probably conflict with it if is installed (this happens if you run this in google colab).
```
python3 -m pip install git+https://github.com/damianoazzolini/pasta
```
Then you can use it as:
```
from pasta import pasta_solver
solver = pasta_solver.Pasta(filename, query)
lp, up = pasp_solver.inference()

print("Lower probability for the query " + query + ": " + str(lp))
print("Upper probability for the query " + query + ": " + str(up))
```
where `filename` and `query` are the name of the file and the query to ask.

You can also pass the file as a string, in this way:
```
from pasta import pasta_solver
solver = pasta_solver.Pasta("", query) # leave the filename empty
lp, up = pasp_solver.inference(file_as_string)
```
where `file_as_string` is a string containing your program.



## Usage

### Exact inference
```
cd src/pasta
python3 pasta_solver.py ../../examples/conditionals/bird_4_cond.lp --query="fly"
```
Asks the query `fly` for the program stored in `../../examples/conditionals/bird_4_cond.lp`.
Expected result:
```
Lower probability for the query: 0.7
Upper probability for the query: 1.0
```

### Abduction
```
cd src/pasta
python3 pasta_solver.py ./../examples/abduction/bird_4_abd_prob.lp --query="fly(1)" --abduction
```

## Syntax
Basically, PASTA (PASP) programs are ASP programs plus probabilistic facts.
Probabilistic facts can be added with the syntax: `prob::atom.` where `prob` is a floating point number (0 < number <= 1) and `atom` is a standard ASP fact.
For example, `0.5::a.` states that `a` has probability `0.5`.

For more examples, see the `examples` folder.

## Description and How to Cite
The papers describing this system will be soon available.