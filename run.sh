#!/bin/bash

# geometry
num_boundary=400
num_eval=2000

mu=10
nu=0.49

python exm_somig.py --num_boundary ${num_boundary} --num_eval ${num_eval} --mu ${mu} --nu ${nu}
