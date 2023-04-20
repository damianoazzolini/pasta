FROM ubuntu:latest

WORKDIR home/pasta

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3-pip

RUN pip install clingo
RUN pip install scipy
RUN pip install numpy

COPY . .

RUN pip install .

# CMD pasta
