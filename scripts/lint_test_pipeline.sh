#!/bin/bash

# echo "--- lint with pylint ---"
# echo "disabled W0311: bad indentation"
# echo "disabled C0103: invalid name"
# echo "disabled E0401: import error"
# echo "disabled C0301: line too long"
# echo "disabled R0913: too many arguments"
# echo "disabled R0914: too many local variables"
# echo "disabled R0902: too many instance variables"

# echo "--- asp_interface.py ---"
# python3 -m pylint ../pasta/asp_interface.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902

# echo "--- generator.py ---"
# python3 -m pylint ../pasta/generator.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902

# echo "--- models_handler.py ---"
# python3 -m pylint ../pasta/models_handler.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902

# echo "--- pasta_parser.py ---"
# python3 -m pylint ../pasta/pasta_parser.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902

# echo "--- pasta_solver.py ---"
# python3 -m pylint ../pasta/pasta_solver.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902

# echo "--- utils.py ---"
# python3 -m pylint ../pasta/utils.py --disable=W0311,C0103,E0401,C0301,R0913,R0914,R0902


# echo "--- lint with flake 8 ---"

# echo "ignored E203: whitespace before ':'"
# echo "ignored E231: whitespace before ':'"
# echo "ignored E123: closing bracket does not match indentation of opening bracket's line"
# echo "ignored E125: continuation line with same indent as next logical line"
# echo "ignored E501: line too long"
# echo "ignored E303: too many blank lines"
# echo "ignored E128: continuation line under-indented for visual indent"
# echo "ignored E124: closing bracket does not match visual indentation"

# flake8 ../pasta/pasta_solver.py --show-source --statistics --ignore=E203,E231,E123,E125,E501,E303
# flake8 ../pasta/asp_interface.py --show-source --statistics --ignore=E203,E231,E123,E125,E501,E303
# flake8 ../pasta/models_handler.py --show-source --statistics --ignore=E203,E231,E123,E125,E501,E303,E128,E124
# flake8 ../pasta/pasta_parser.py --show-source --statistics --ignore=E203,E231,E123,E125,E501,E303,E128,E124

# echo "-- test with pytest ---"
# cd .. && cd tests && pytest

echo '--- prospector ---'
prospector ../pasta/ -s verylow

# p3 -m black <file> : refactor automatico

# p3 bowler

# p3 -m pip install --editable .