#!/bin/bash
python evaluator.py --subnet ./net_1.subnet --data ../../data/LC25000 --dataset LC25000 --n_classes 5 --supernet_path ./model_best_FL_w1.0 --save net_1_FR_noFL.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 200 --resolution 224 --str_time 20211004-214637-1-FR-noFL --reset_running_statistics --flag_fuzzy
