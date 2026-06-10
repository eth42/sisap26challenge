#!/bin/bash

docker build -t sisap26 .
sudo su root -c "echo 3 > /proc/sys/vm/drop_caches"
