#!/bin/bash

cd build/

case=(2 4 6 8 10)
for i in ${case[@]}
  do echo nq=$i; CUDA_VISIBLE_DEVICES=1 ./benchmark05 ${i} ${i} ${i} &> ../nq${i}x${i}x${i}.log
done
