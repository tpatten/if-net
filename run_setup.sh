#!/bin/bash

# Convert to .off format
python3.7 data_processing/convert_to_scaled_off.py

# Create input data
python3.7 data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300

# Training input and occupancy values
python3.7 data_processing/boundary_sampling.py -sigma 0.1
python3.7 data_processing/boundary_sampling.py -sigma 0.01

# Remove bad meshes
# python3.7 data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete

# For visualization
python3.7 data_processing/create_pc_off.py -res 128 -num_points 300

# Training
#python3.7 train.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -batch_size 6 -pointcloud -pc_samples 3000

# Generation
#python3.7 generate.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 12 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300

# python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 12 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300 -datapath 02828884/1a40eaf5919b1b3f3eaa2b95b99dae6
