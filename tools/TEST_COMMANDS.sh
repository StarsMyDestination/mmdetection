#!/usr/bin/env bash

## ============== dataset converter ================##
python dataset_converters/cityscapes.py /mnt/nas/DATA/CityScapes


#====================== nvidia DLProf =========================#
dlprof -f true --mode=pytorch --reports=summary,detail,iteration,kernel,tensor \
--delay 10 --duration 10 --output_path=./nsys python test.py --use_profile True

# use key_node to separate iteration
dlprof -f true --mode=pytorch --key_node=TOPK_1 --reports=summary,detail,iteration,kernel,tensor \
--delay 10 --duration 10 --output_path=./nsys python test.py --use_profile True