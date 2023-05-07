#!/usr/bin/env bash
DIR=$(dirname "$(readlink -f "$0")")

#NAME=$(basename $1 .py)
EXPDIR=$DIR/log
singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/run_eval.py $DIR --mode test
