FROM python:3

WORKDIR home/pasta

# COPY <src> <dest>
COPY examples examples/

COPY src src/

RUN pip install clingo
RUN pip install scipy