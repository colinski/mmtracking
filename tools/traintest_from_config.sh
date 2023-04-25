#!/usr/bin/env bash
bash $WORK/src/mmtracking/tools/train_from_config.sh $1 $2 
bash $WORK/src/mmtracking/tools/test_from_config.sh $1 1
