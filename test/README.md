# How to run tests
To run all tests:
```
python3 test.py
```

To test a specific module:
```
python3 test_module_name.py
# for example
python3 test_utilities.py
```

Note: python3.5 is required (for type hint for arguments of functions and testing).

For coverage: python3 -m pip install coverage
Then,
coverage run -m unittest test_pasta_parser.py
and 
coverage html
to generate html 
or 
coverage report -m
to have the results printed on the terminal.