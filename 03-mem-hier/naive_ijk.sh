#!/bin/bash

for ((n=511;n<=513;n++))
do
    for k in ijk jik ikj kij jki kji
    do
        ./matrix_prod -n $k $n $n $n
    done
done
