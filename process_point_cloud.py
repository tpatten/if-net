import os
import trimesh
import argparse
import numpy as np
import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree


def convert_to_scaled_off(filename):
    # Convert .obj (or .ply) to off format
    _, file_extension = os.path.splitext(filename)
    if file_extension == 'obj':
        output_file = filename.replace('.obj', '.off')
    elif file_extension == '.ply':
        output_file = filename.replace('.ply', '.off')
    else:
        raise Exception('Can not convert file of type {}'.format(file_extension))

    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(filename, output_file)
    cmd = 'meshlabserver -i {} -o {}'.format(filename, output_file)
    os.system(cmd)

    # Scale the .off file
    try:
        mesh = trimesh.load(output_file, process=False)
        offset = np.expand_dims(np.mean(mesh.vertices, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(mesh.vertices ** 2, axis=1)), 0)

        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        #centers = offset.reshape((3,))
        #total_size = dist

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        output_file = output_file.replace('.off', '')
        output_file += '_scaled.off'
        mesh.export(output_file)
    except:
        raise Exception('Error with {}'.format(filename))

    return mesh


def voxelized_pointcloud_sampling(dirpath, mesh):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the point cloud to usable format for IF-Net')
    parser.add_argument('-filename', type=str)
    args = parser.parse_args()

    dirpath = os.path.dirname(args.filename)

    scaled_cloud = convert_to_scaled_off(args.filename)
    voxelized_cloud_filename = voxelized_pointcloud_sampling(dirpath, scaled_cloud)
    sigmas = [0.1, 0.01]
    for s in sigmas:
        boundary_sampling(dirpath, scaled_cloud, s)
