EXPDIR=exps/two_truck_node_14_5p0f_meters
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_train.sh $EXPDIR/config.py 1 --work-dir $EXPDIR/log --seed 5
