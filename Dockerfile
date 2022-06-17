# parto da un'immagine esistente che ha un ambiente di 
# lavoro con già installato Maven e Java 11: https://hub.docker.com/_/maven
FROM python:3

WORKDIR home/pasta

# copio il file pom.xml nella WORKDIR
# COPY <src> <dest>
COPY examples examples/

# copio la cartella src nella WORKDIR
# importante specificare src/ così crea la
# cartella e non copia solamente il contenuto
COPY src src/

RUN pip install clingo