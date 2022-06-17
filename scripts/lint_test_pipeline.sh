#!/bin/bash

echo "--- lint with pylint ---"
echo "disabled W0311: bad indentation"
python3 -m pylint ../src/pasta --disable=W0311

echo "--- lint with flake 8 ---"
flake8 ../ --show-source --statistics

echo "-- test with pytest ---"
cd .. && cd tests && pytest