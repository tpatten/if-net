import os
import trimesh
import argparse
import numpy as np
import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
import traceback
import json


def convert_to_scaled_off(filename):
    # Convert .obj (or .ply) to off format
    _, file_extension = os.path.splitext(filename)
    if file_extension == 'obj':
        out_file = filename.replace('.obj', '.off')
    elif file_extension == '.ply':
        out_file = filename.replace('.ply', '.off')
    else:
        raise Exception('Can not convert file of type {}'.format(file_extension))

    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(filename, out_file)
    cmd = 'meshlabserver -i {} -o {}'.format(filename, out_file)
    os.system(cmd)

    # Scale the .off file
    try:
        mesh = trimesh.load(out_file, process=False)
        # offset = np.expand_dims(np.mean(mesh.vertices, axis=0), 0)
        # dist = np.max(np.sqrt(np.sum(mesh.vertices ** 2, axis=1)), 0)

        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
        print('Total size: {}'.format(total_size))
        print('Centers: {}'.format(centers))

        # centers = offset.reshape((3,))
        # total_size = dist

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        #out_file = out_file.replace('.off', '')
        #out_file += '_scaled.off'
        out_file = os.path.join(os.path.dirname(args.filename), 'mesh_scaled.off')
        mesh.export(out_file)

        scale_data = {'total_size': total_size, 'centers': list(centers)}
        out_file = os.path.join(os.path.dirname(args.filename), 'scale_params.json')
        with open(out_file, 'w') as file:
            json.dump(scale_data, file, indent=2)
    except:
        raise Exception('Error with {}'.format(filename))

    return mesh


'''
def convert_to_scaled_off2(filename):
    # Convert .obj (or .ply) to off format
    _, file_extension = os.path.splitext(filename)
    if file_extension == 'obj':
        out_file = filename.replace('.obj', '.ply')
    elif file_extension == '.ply':
        out_file = filename.replace('.ply', '.ply')
    else:
        raise Exception('Can not convert file of type {}'.format(file_extension))

    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(filename, out_file)
    # cmd = 'meshlabserver -i {} -o {}'.format(filename, out_file)
    # os.system(cmd)

    # Load the point cloud
    cloud = o3d.io.read_point_cloud(filename)

    # Get the points as numpy
    pts = np.asarray(cloud.points)

    # Scale and points
    bounds = [cloud.get_max_bound(), cloud.get_min_bound()]
    total_size = (bounds[1] - bounds[0]).max()
    centers = (bounds[1] + bounds[0]) / 2
    pts = pts - centers
    pts = pts / np.abs(total_size)

    # Save as mesh
    cloud.points = o3d.utility.Vector3dVector(pts)
    out_file = os.path.join(os.path.dirname(args.filename), 'mesh_scaled.ply')
    o3d.io.write_point_cloud(out_file, cloud)

    return cloud
'''


def voxelized_pointcloud_sampling(dirpath, mesh, num_points):
    # Variables (do not change)
    res = 128
    bb_min = -0.5
    bb_max = 0.5

    # Create the grid points
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)

    # Create kdtree
    kdtree = KDTree(grid_points)

    # Generate a point cloud from the mesh
    point_cloud = mesh.sample(num_points)

    # Compute the occupancies
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(point_cloud)
    occupancies[idx] = 1
    compressed_occupancies = np.packbits(occupancies)

    # Save as .npz
    out_file = os.path.join(dirpath, 'voxelized.npz')
    np.savez(out_file, point_cloud=point_cloud, compressed_occupancies=compressed_occupancies, bb_min=bb_min,
             bb_max=bb_max, res=res)

    # Save cloud for visualization
    pc = np.load(out_file)['point_cloud']
    out_file = out_file.replace('.npz', '.off')
    trimesh.Trimesh(vertices=pc, faces=[]).export(out_file)

    return out_file


'''
def voxelized_pointcloud_sampling2(dirpath, mesh):
    # Variables (do not change)
    res = 128
    num_points = 3000
    bb_min = -0.5
    bb_max = 0.5

    # Create the grid points
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)

    # Create kdtree
    kdtree = KDTree(grid_points)

    # Generate a point cloud from the mesh
    # point_cloud = mesh.sample(num_points)

    pts = np.asarray(mesh.points)
    choice = np.random.choice(pts.shape[0], num_points, replace=True)
    pts = pts[choice, :]

    # Compute the occupancies
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(pts)
    occupancies[idx] = 1
    compressed_occupancies = np.packbits(occupancies)

    # Save as .npz
    out_file = os.path.join(dirpath, 'voxelized.npz')
    np.savez(out_file, point_cloud=pts, compressed_occupancies=compressed_occupancies, bb_min=bb_min,
             bb_max=bb_max, res=res)

    # Save cloud for visualization
    out_file = out_file.replace('.npz', '.ply')
    # trimesh.Trimesh(vertices=pc, faces=[]).export(out_file)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(out_file, cloud)

    return out_file
'''


def boundary_sampling(dirpath, mesh, sigma):
    try:
        # Get the points
        sample_num = 100000
        points = mesh.sample(sample_num)

        # Generate boundary points
        boundary_points = points + sigma * np.random.randn(sample_num, 3)

        # Generate grid points
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
        grid_coords = 2 * grid_coords

        # Generate occupancies
        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        # Save as .npz
        out_file = os.path.join(dirpath, 'boundary_{}_samples.npz'.format(sigma))
        np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
    except:
        raise Exception('Error with {}: {}'.format(dirpath, traceback.format_exc()))

    return out_file

'''
def boundary_sampling2(dirpath, mesh, sigma):
    try:
        # Get the points
        sample_num = 100000
        # points = mesh.sample(sample_num)
        points = np.asarray(mesh.points)
        choice = np.random.choice(points.shape[0], sample_num, replace=True)
        points = points[choice, :]

        # Generate boundary points
        boundary_points = points + sigma * np.random.randn(sample_num, 3)

        # Generate grid points
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
        grid_coords = 2 * grid_coords

        # Generate occupancies
        tmesh = trimesh.Trimesh(vertices=np.asarray(mesh.points), faces=[])
        occupancies = iw.implicit_waterproofing(tmesh, boundary_points)[0]

        # Save as .npz
        out_file = os.path.join(dirpath, 'boundary_{}_samples.npz'.format(sigma))
        np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
    except:
        raise Exception('Error with {}: {}'.format(dirpath, traceback.format_exc()))

    return out_file
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the point cloud to usable format for IF-Net')
    parser.add_argument('-filename', type=str)
    parser.add_argument('-pc_samples', default=3000, type=int)
    args = parser.parse_args()

    dirpath = os.path.dirname(args.filename)

    scaled_cloud = convert_to_scaled_off(args.filename)
    voxelized_cloud_filename = voxelized_pointcloud_sampling(dirpath, scaled_cloud, args.pc_samples)
    sigmas = [0.1, 0.01]
    for s in sigmas:
        boundary_sampling(dirpath, scaled_cloud, s)
