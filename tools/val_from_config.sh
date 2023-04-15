#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

mkdir -p $EXPDIR/val

singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval track vid --cfg-options evaluation.logdir=$EXPDIR/val evaluation.video_length=5 evaluation.dataset=val evaluation.grid_search=True
