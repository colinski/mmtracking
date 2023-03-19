#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

mkdir -p $EXPDIR/export

#singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_export.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval track vid --cfg-options evaluation.logdir=$EXPDIR/export evaluation.video_length=500 evaluation.dataset=test evaluation.grid_search=False evaluation.calib_file=$EXPDIR/val/res.json evaluation.calib_metric=nll
