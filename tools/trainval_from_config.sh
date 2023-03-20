#!/usr/bin/env bash

CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME
VALDIR=$EXPDIR/val
#if [ -d $EXPDIR ]; then
    #rm -rf $EXPDIR/*
#fi
#--cfg-options "model=dict(init_cfg=dict(type='Pretrained',checkpoint='${EXPDIR}/latest.pth'))"
singularity run --nv -H $WORK $WORK/sif/python.sif python $WORK/src/mmtracking/tools/cache_datasets.py $CONFIG
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_train.sh $CONFIG $2\
    --work-dir $VALDIR\
    --seed 5\
    --trainset val\
    --cfg-options "model.init_cfg.type=Pretrained" "model.init_cfg.checkpoint=${EXPDIR}/latest.pth" "model.cov_only_train=True" #"total_epochs=10"
