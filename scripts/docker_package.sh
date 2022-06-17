#!/bin/bash

VERSION=0.0.1

# create the image
docker image build --tag pasta:$VERSION ../

# to run
# docker container run -it pasta:$VERSION bash

# associate the image
docker image tag pasta:$VERSION damianodamianodamiano/pasta:$VERSION

# push the image
docker image push damianodamianodamiano/pasta:$VERSION