#!/bin/bash

module load cuda

cd build/

case=(2 4 6 8 10 12 14 16 32)
for i in ${case[@]}
  do echo nq=$i; CUDA_VISIBLE_DEVICES=1 ./benchmark04  ${i} ${i} &> ../nq${i}x${i}.log
done
