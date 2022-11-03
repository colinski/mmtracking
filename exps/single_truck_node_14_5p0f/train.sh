EXPDIR=exps/single_truck_node_14_5p0f
singularity run --nv -H $WORK $WORK/sif/python_uncv.sif $WORK/src/mmtracking/tools/dist_train.sh $EXPDIR/config.py 1 --work-dir $EXPDIR/log --seed 5
