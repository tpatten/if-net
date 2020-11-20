#!/bin/bash

# Convert to .off format
#python3.7 data_processing/convert_to_scaled_off.py

# Create input data
#python3.7 data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300

# Training input and occupancy values
#python3.7 data_processing/boundary_sampling.py -sigma 0.1
#python3.7 data_processing/boundary_sampling.py -sigma 0.01

# Remove bad meshes
# python3.7 data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete

# For visualization
#python3.7 data_processing/create_pc_off.py -res 128 -num_points 300

# Training
#python3.7 train.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -batch_size 6 -pointcloud -pc_samples 3000

# Generation
#python3.7 generate.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 12 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300

# python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 12 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300 -datapath 02828884/1a40eaf5919b1b3f3eaa2b95b99dae6



#SUBJS=(ABF10 BB10 GPMF10 MC1 MDF10 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
#RECONS=(ball_pivot tsdf poisson)

#for r in "${RECONS[@]}"
#do
#    for s in "${SUBJS[@]}"
#    do
#        #python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 12 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300 -datapath ho3d/ho3d_Segmentation_step0-3_300/$s/$r
#        python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -checkpoint 23 -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 3000 -datapath ho3d/ho3d_Segmentation_step0-3_3000/$s/$r
#        #python3.7 process_point_cloud.py -file /home/tpatten/Code/if-net/shapenet/data/ho3d/ho3d_Segmentation_step0-3_3000/$s/$r/mesh.ply
#    done
#done


#SUBJS=(ABF12 BB12 GPMF12 GSF12 MC1 MC4 MDF12 SB12 ShSu12 SiBF12 SM2 SM3 SMu1 SMu40 SS1)
SUBJS=(GPMF12)
for s in "${SUBJS[@]}"
do
	python3.7 process_point_cloud.py -filename /home/tpatten/Code/if-net/shapenet/data/ho3d/obj_learn/${s}_300/mesh.ply -pc_samples 300
	python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 300 -datapath ho3d/obj_learn/${s}_300
    python3.7 process_point_cloud.py -filename /home/tpatten/Code/if-net/shapenet/data/ho3d/obj_learn/${s}_3000/mesh.ply -pc_samples 3000
	python3.7 demo.py -std_dev 0.1 0.01 -res 128 -m ShapeNetPoints -batch_points 200000 -std_dev 0.1 0.01 -pointcloud -pc_samples 3000 -datapath ho3d/obj_learn/${s}_3000
done
