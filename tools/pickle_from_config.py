#!/usr/bin/env bash

CONFIG=$1
#NAME=$(basename $1 .py)
#EXPDIR=logs/$NAME
#if [ -d $EXPDIR ]; then
    #rm -rf $EXPDIR/*
#fi
singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/pickle_datasets.py $CONFIG
