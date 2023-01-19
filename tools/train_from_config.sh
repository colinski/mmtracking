#!/usr/bin/env bash

CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME
if [ -d $EXPDIR ]; then
    rm -rf $EXPDIR/*
fi
singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_train.sh $CONFIG $2 --work-dir $EXPDIR --seed 5
