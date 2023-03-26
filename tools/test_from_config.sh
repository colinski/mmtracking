#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG

mkdir -p $EXPDIR/train
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/train evaluation.video_length=500 evaluation.dataset=train evaluation.grid_search=False
mkdir -p $EXPDIR/val
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/val evaluation.video_length=500 evaluation.dataset=val evaluation.grid_search=False
mkdir -p $EXPDIR/test
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/test evaluation.video_length=500 evaluation.dataset=test evaluation.grid_search=False
