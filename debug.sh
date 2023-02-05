singularity run --nv ~/sif/python.sif python test.py

#singularity run ~/sif/open-mmlab:1.0.sif python tools/convert_datasets/mot/mot2coco.py -i $HOME/data/MOT15/ -o $HOME/data/MOT15/annotations --convert-det

#singularity run ~/sif/open-mmlab:1.0.sif python tools/convert_datasets/mot/mot2coco.py -i $HOME/data/MOT20/ -o $HOME/data/MOT20/annotations --split-train --convert-det

#singularity run ~/sif/open-mmlab:1.0.sif python tools/convert_datasets/mot/mot2coco.py -i $HOME/data/MOT20/ -o $HOME/data/MOT20/annotations --convert-det
