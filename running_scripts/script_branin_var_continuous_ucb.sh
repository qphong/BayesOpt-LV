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

WIDTH=0.1
NZSAMPLE=50
NXSAMPLE=50
NTRAINSUR=1000

MINVAR=1e-4

python ../run\_var\_UCB\_continuous.py --gpu $GPU --function $FUNCTION --numqueries $NUMQUERIES --numruns $NUMRUNS  --ntrain $NTRAIN --n\_init\_data $NINITDATA --n_rand_opt_init 2 --quantile $QUANTILE --n\_iter\_fitgp $NITERGP --minvar $MINVAR --dtype $DTYPE --width $WIDTH --nzsample $NZSAMPLE --nxsample $NXSAMPLE --ntrainsur $NTRAINSUR


