#!/bin/bash

conda activate mrna

num=("max_gen: 1000" "max_gen: 2000" "max_gen: 3000" "max_gen: 4000" "max_gen: 5000")
type="protein"
seq="AUGAACGAUACGGAAGCGAUC"
for i in "${num[@]}"
do
    sed -i "4s/.*/$i/" ./testConfig/mRNA.yaml
    pid=$(nohup python DE.py ${type} ${seq} </dev/null &>/dev/null & echo $!)
    echo "${pid}: ${i}"
    while true
    do
      if ps -p $pid > /dev/null
      then
        sleep 60
      else
        break
      fi
    done
done
```