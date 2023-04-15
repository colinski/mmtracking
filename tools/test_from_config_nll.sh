#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

mkdir -p $EXPDIR/test_nll

singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval track vid --cfg-options evaluation.logdir=$EXPDIR/test_nll evaluation.video_length=5 evaluation.dataset=test evaluation.grid_search=False evaluation.calib_file=$EXPDIR/val/res.json evaluation.calib_metric=nll
