#! /bin/bash

GPU=0
FUNCTION=negative\_branin\_uniform
NUMRUNS=10
NUMQUERIES=40
NTRAIN=2000
NINITDATA=3
QUANTILE=0.1
NITERGP=1
DTYPE=float64

MINVAR=1e-4

BETA=3.0
NFEATURE=200

declare -a zmodes=('prob')

# number of Thompson sampling samples
NTSSAMPLE=1

for ((i=0;i<${#zmodes[@]};++i));
do
    python ../run\_cvar\_TS\_discrete.py --gpu $GPU --function $FUNCTION --numqueries $NUMQUERIES --numruns $NUMRUNS  --ntrain $NTRAIN --n\_init\_data $NINITDATA --n_rand_opt_init 2 --beta $BETA --nthompsonsample $NTSSAMPLE --nfeature $NFEATURE --quantile $QUANTILE --n\_iter\_fitgp $NITERGP --minvar $MINVAR --zmode "${zmodes[i]}" --dtype $DTYPE
done