#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG

#mkdir -p $EXPDIR/train
#singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/train evaluation.video_length=500 evaluation.dataset=train evaluation.grid_search=False

mkdir -p $EXPDIR/val1
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/val1 evaluation.dataset=val1 evaluation.grid_search=False

mkdir -p $EXPDIR/val2
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/val2 evaluation.dataset=val2 evaluation.grid_search=False

mkdir -p $EXPDIR/test1
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/test1 evaluation.dataset=test1 evaluation.grid_search=False

mkdir -p $EXPDIR/test2
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/test2 evaluation.dataset=test2 evaluation.grid_search=False


#mkdir -p $EXPDIR/test
#singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval vid --cfg-options evaluation.logdir=$EXPDIR/test evaluation.video_length=500 evaluation.dataset=test evaluation.grid_search=False
