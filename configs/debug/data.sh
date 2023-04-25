singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Bus_empty_1_6D/train/ /dev/shm/Bus_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Bus_empty_1_6D/val/ /dev/shm/Bus_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Bus_empty_1_6D/test/ /dev/shm/Bus_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Car_empty_1_6D/train/ /dev/shm/Car_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Car_empty_1_6D/val/ /dev/shm/Car_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/Car_empty_1_6D/test/ /dev/shm/Car_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/All_vehicle_1_6D/train/ /dev/shm/All_vehicle_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/All_vehicle_1_6D/val/ /dev/shm/All_vehicle_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py $WORK/data_405/All_vehicle_1_6D/test/ /dev/shm/All_vehicle_1_6D_test --valid_mods mocap zed_camera_left --overwrite
