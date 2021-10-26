# BayesOpt-LV
Optimizing Value-at-Risk and Conditional Value-at-Risk of Black Box Functions with Lacing Values (LV)

## About
This repository contains the source code for 
* ICML'21 paper: *Value-at-Risk Optimization with Gaussian Processes*
* NeurIPS'21 paper: *Optimizing Conditional Value-At-Risk of Black-Box Functions*

## Requirements
```
numpy
scipy
tensorflow 1.14.0
tensorflow-probability 0.7.0
gpflow 1.5.1
```

## Instructions

The examples of running scripts are in `running_scripts` folder. 
The optimization results are stored in a folder named with the objective function in `running_scripts`.
Objective functions are found in `functions.py`.