#!/bin/bash

docker rmi $(docker images --filter=reference="sisap26challenge-*" -q)
