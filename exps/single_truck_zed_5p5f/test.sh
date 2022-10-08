EXPDIR=exps/single_truck_zed_5p5f
singularity run --nv -H $WORK $WORK/sif/python_uncv.sif $WORK/src/mmtracking/tools/dist_test.sh $EXPDIR/config.py 1 --checkpoint $EXPDIR/log/latest.pth --eval track

mpv $EXPDIR/latest_vid.mp4
