#!/bin/bash

make cleanall
for i in `seq 0 $1` 
do
    echo "----------- running K = $((10**i))"
    make runall k=$((10**i))
done
make cleanall