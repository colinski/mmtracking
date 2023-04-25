DATA_ROOT=~/eight/data_405/scenarios
#DATA_ROOT=$WORK/data_405

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Bus_empty_1_6D/train/ /dev/shm/Bus_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Bus_empty_1_6D/val/ /dev/shm/Bus_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Bus_empty_1_6D/test/ /dev/shm/Bus_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Car_empty_1_6D/train/ /dev/shm/Car_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Car_empty_1_6D/val/ /dev/shm/Car_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/Car_empty_1_6D/test/ /dev/shm/Car_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/All_vehicle_1_6D/train/ /dev/shm/All_vehicle_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/All_vehicle_1_6D/val/ /dev/shm/All_vehicle_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $DATA_ROOT/All_vehicle_1_6D/test/ /dev/shm/All_vehicle_1_6D_test --valid_mods mocap zed_camera_left --overwrite
