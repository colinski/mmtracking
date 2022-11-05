EXPDIR=exps/oracle
singularity run --nv -H $WORK $WORK/sif/python.sif $WORK/src/mmtracking/tools/dist_test.sh $EXPDIR/config.py 1 --eval track
mpv $EXPDIR/latest_vid.mp4
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_test.sh src/mmtracking/configs/mocap/single_truck_901_zed_5x5.py 1 --checkpoint $LOGDIR/latest.pth --eval track
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_test.sh src/mmtracking/configs/mocap/single_truck_901_zed_5x5.py 1 --checkpoint $LOGDIR/latest.pth --eval track

#LOGDIR=logs/single_truck_901_zed
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_test.sh src/mmtracking/configs/mocap/single_truck_901_zed.py 1 --checkpoint $LOGDIR/latest.pth --eval track
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_train.sh src/mmtracking/configs/mocap/single_truck_901_zed_time1.py 1 --work-dir $LOGDIR --seed 5

#mv logs/single_truck/latest_vid.mp4 $LOGDIR

#LOGDIR=logs/single_truck_901_zed_range_v2
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_train.sh src/mmtracking/configs/mocap/single_truck_901_zed_range.py 1 --work-dir $LOGDIR --seed 5
#mv logs/single_truck/latest_vid.mp4 $LOGDIR

#LOGDIR=logs/single_truck_901_zed_v2
#singularity run --nv -H $WORK sif/python_uncv.sif src/mmtracking/tools/dist_train.sh src/mmtracking/configs/mocap/single_truck_901_zed.py 1 --work-dir $LOGDIR --seed 5
#mv logs/single_truck/latest_vid.mp4 $LOGDIR





#singularity run --nv sif/python_uncv.sif src/mmtracking/tools/dist_test.sh src/mmtracking/configs/mocap/two_truck_901.py 1 --checkpoint $LOGDIR/latest.pth --eval track
#LOGDIR=logs/two_trucks_lr=4e4_100e_mse0.1
#singularity run --nv sif/python_uncv.sif src/mmtracking/tools/dist_train.sh src/mmtracking/configs/mocap/two_truck.py 1 --work-dir $LOGDIR --seed 5

#singularity run --nv sif/python_uncv.sif src/mmtracking/tools/dist_test.sh src/mmtracking/configs/mocap/two_truck.py 1 --checkpoint $LOGDIR/latest.pth --eval track

#mpv logs/two_trucks/latest_vid.mp4
