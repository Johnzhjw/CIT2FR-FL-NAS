#!/bin/bash
module load anaconda/2020.11
source activate torch1.4
python msunas.py --data ../data/LC25000 --dataset LC25000 --n_classes 5 --supernet_path ./model_best_FL_w1.0 --save search-init --sec_obj flops --n_gpus 1 --n_workers 4 --n_epochs 0 --flag_FL --size_FL 3
