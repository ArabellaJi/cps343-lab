#!/bin/bash

for ((n=1;n<=3;n++))
do
    for x in $(seq -w 2.1 0.2 2.9)
    do
        for ((m=500000;m<=600000;m+=10000))
        do
            ./args $n $x $m
        done
    done
done
