#!/usr/bin/env bash

CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/log/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR


