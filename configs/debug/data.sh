singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Bus_empty_1_6D/train/ ~/Bus_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Bus_empty_1_6D/val/ ~/Bus_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Bus_empty_1_6D/test/ ~/Bus_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Car_empty_1_6D/train/ ~/Car_empty_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Car_empty_1_6D/val/ ~/Car_empty_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/Car_empty_1_6D/test/ ~/Car_empty_1_6D_test --valid_mods mocap zed_camera_left --overwrite

singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/All_vehicle_1_6D/train/ ~/All_vehicle_1_6D_train --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/All_vehicle_1_6D/val/ ~/All_vehicle_1_6D_val --valid_mods mocap zed_camera_left --overwrite
singularity run --nv sif/python.sif python src/mmtracking/tools/pickle_datasets.py eight/data_405/scenarios/All_vehicle_1_6D/test/ ~/All_vehicle_1_6D_test --valid_mods mocap zed_camera_left --overwrite
