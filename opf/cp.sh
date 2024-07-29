#!/bin/bash
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_19/opf/two33gen15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_19/opf/two33load15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_17/opf/two69gen15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_17/opf/two69load15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_19/opf/two118gen15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/
cp -r "/home/liuqiong/software/OSTC_partical_1min_static_state_19/opf/two118load15.npy" /home/liuqiong/software/OSTC_partical_1min_static_state_19/


python MBO15.py --env_name 33 &
python MBO15.py --env_name 69 &
python MBO15.py --env_name 118 &