#!/usr/bin/env bash
DIR=$(dirname "$(readlink -f "$0")")

#NAME=$(basename $1 .py)
EXPDIR=$DIR/log
if [ -d $EXPDIR ]; then
    rm -rf $EXPDIR/*
fi
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_train.sh $DIR/config $1--work-dir $EXPDIR --seed 5
