# mmtracking

This repo began as a fork of [open-mmlab's mmtracking repo](https://github.com/open-mmlab/mmtracking). It has been heavily modified and most of the original 
code (dealing with bounding box tracking) has been removed.

## Work Environment
In this section I describe how I have my working enviroment setup.
I have a `$WORK` environment variable defined in my bashrc which defines the location where everything is stored. On my local machine it is just `$HOME`.
On Unity it has to be set to `/work/$USER`.  

In `$WORK` I have a directory  `src` which contains all the cloned repos that I need. We can get access to them via:
```
cd src/
git clone https://github.com/colinski/mmclassification.git
git clone https://github.com/colinski/mmdetection.git
git clone https://github.com/colinski/mmtracking.git
git clone https://github.com/colinski/TrackEval.git
git clone https://github.com/colinski/yolov7.git
git clone https://github.com/reml-lab/resource_constrained_tracking
git clone https://github.com/shiwei-fang/resource-allocation
```
I then add all them to my PYTHONPATH in my bashrc 
```
export PYTHONPATH=$PYTHONPATH:$WORK/src/mmdetection
export PYTHONPATH=$PYTHONPATH:$WORK/src/mmclassification
export PYTHONPATH=$PYTHONPATH:$WORK/src/mmtracking
export PYTHONPATH=$PYTHONPATH:$WORK/src/TrackEval
export PYTHONPATH=$PYTHONPATH:$WORK/src/resource_constrained_tracking
export PYTHONPATH=$PYTHONPATH:$WORK/src/yolov7
export PYTHONPATH=$PYTHONPATH:$WORK/src/resource-allocation
```

These are repos that contain code that I have changed or may have to change in the future. The rest of enviroment (Torch, NumPy, etc.) is containerized.
The dockerfiles can be found at ``github.com/colinski/dockerfiles.git``, but it would be easier for me to transfer the image directly rather than rebuilding.
On the Unity cluster we do not have root access, so to use containers we have to use [singularity](https://docs.sylabs.io/guides/latest/user-guide/). I build docker images using a standard Dockerfile and then 
convert them to singularity's `.sif` format. 

An example singularity is as follows:
```
singularity run --nv -H $WORK $WORK/sif/python.sif python
```
The `--nv` option enables access to aviaible GPUs, `-H $WORK` sets the home directionary of the container to be `$WORK`, `$WORK/sif/python.sif` is the path to the
image file, and finally `python` is the command to be run.


