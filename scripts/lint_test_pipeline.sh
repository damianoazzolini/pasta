#!/bin/bash

echo "--- lint with pylint ---"
echo "disabled W0311: bad indentation"
python3 -m pylint ../src/pasta --disable=W0311,C0103

# echo "--- lint with flake 8 ---"
# echo "ignored E203: whitespace before ':'"
# echo "ignored E123: closing bracket does not match indentation of opening bracket's line"
# echo "ignored E125: continuation line with same indent as next logical line"
# flake8 ../ --show-source --statistics --ignore=E203,E123,E125

# echo "-- test with pytest ---"
# cd .. && cd tests && pytest