EXPDIR=exps/single_truck_node_1_5p0f_meters
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $EXPDIR/config.py 1 --checkpoint $EXPDIR/log/latest.pth --eval track
mpv $EXPDIR/latest_vid.mp4
