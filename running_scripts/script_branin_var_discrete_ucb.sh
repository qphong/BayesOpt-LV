#! /bin/bash

GPU=0
FUNCTION=negative\_branin\_uniform
NUMRUNS=10
NUMQUERIES=60
NTRAIN=2000
NINITDATA=3
QUANTILE=0.1
NITERGP=3
DTYPE=float64

MINVAR=1e-4

declare -a zmodes=('prob')

for ((i=0;i<${#zmodes[@]};++i));
do
    python ../run\_var\_UCB\_discrete.py --gpu $GPU --function $FUNCTION --numqueries $NUMQUERIES --numruns $NUMRUNS  --ntrain $NTRAIN --n\_init\_data $NINITDATA --n_rand_opt_init 2 --quantile $QUANTILE --n\_iter\_fitgp $NITERGP --minvar $MINVAR --zmode "${zmodes[i]}" --dtype $DTYPE
done

