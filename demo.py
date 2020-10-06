import os
import models.local_model as model
# import models.data.voxelized_data_shapenet as voxelized_data
import numpy as np
import argparse
from models.generation import Generator
# from generation_iterator import gen_iterator
import torch


SHAPENET_PATH = '/home/tpatten/Code/if-net/shapenet/data/'


def load_data(datapath, res, pointcloud_samples, num_samples, sample_sigmas, voxelized_pointcloud=True):
    path = os.path.join(SHAPENET_PATH, datapath)
    if not voxelized_pointcloud:
        occupancies = np.load(path + '/voxelization_{}.npy'.format(res))
        occupancies = np.unpackbits(occupancies)
        input = np.reshape(occupancies, (res,) * 3)
    else:
        # voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(res, pointcloud_samples)
        voxel_path = path + '/voxelized.npz'
        occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
        input = np.reshape(occupancies, (res,) * 3)

    points = []
    coords = []
    occupancies = []

    for i, num in enumerate(num_samples):
        # boundary_samples_path = path + '/boundary_{}_samples.npz'.format(sample_sigmas[i])
        boundary_samples_path = path + '/boundary_{}_samples.npz'.format(sample_sigmas[i])
        boundary_samples_npz = np.load(boundary_samples_path)
        boundary_sample_points = boundary_samples_npz['points']
        boundary_sample_coords = boundary_samples_npz['grid_coords']
        boundary_sample_occupancies = boundary_samples_npz['occupancies']
        subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
        points.extend(boundary_sample_points[subsample_indices])
        coords.extend(boundary_sample_coords[subsample_indices])
        occupancies.extend(boundary_sample_occupancies[subsample_indices])

    assert len(points) == num_sample_points
    assert len(occupancies) == num_sample_points
    assert len(coords) == num_sample_points

    # Convert to tensors
    grid_coords = np.array(coords, dtype=np.float32)
    grid_coords_tensor = np.zeros((1, grid_coords.shape[0], grid_coords.shape[1]), dtype=np.float32)
    grid_coords_tensor[0, :, :] = grid_coords
    occupancies = np.array(occupancies, dtype=np.float32)
    occupancies_tensor = np.zeros((1, occupancies.shape[0]), dtype=np.float32)
    occupancies_tensor[0, :] = occupancies
    points = np.array(points, dtype=np.float32)
    points_tensor = np.zeros((1, points.shape[0], points.shape[1]), dtype=np.float32)
    points_tensor[0, :, :] = points
    input = np.array(input, dtype=np.float32)
    input_tensor = np.zeros((1, input.shape[0], input.shape[1], input.shape[2]), dtype=np.float32)
    input_tensor[0, :, :, :] = input

    # Return
    return {'grid_coords': torch.from_numpy(grid_coords_tensor),
            'occupancies': torch.from_numpy(occupancies_tensor),
            'points': torch.from_numpy(points_tensor),
            'inputs': torch.from_numpy(input_tensor),
            'path': path}


def generate_mesh(gen, data):
    logits = gen.generate_mesh(data)
    # data_tupels = []
    # data_tupels.append((logits, data, out_path))
    # data_tupels.append((logits, data))
    mesh = gen.mesh_from_logits(logits)

    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run demo')
    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-pc_samples', default=3000, type=int)
    parser.add_argument('-dist', '--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
    parser.add_argument('-std_dev', '--sample_sigmas', default=[], nargs='+', type=float)
    parser.add_argument('-res', default=32, type=int)
    parser.add_argument('-decoder_hidden_dim', default=256, type=int)
    parser.add_argument('-mode', default='test', type=str)
    parser.add_argument('-retrieval_res', default=256, type=int)
    parser.add_argument('-checkpoint', type=int)
    parser.add_argument('-batch_points', default=1000000, type=int)
    parser.add_argument('-m', '--model', default='LocNet', type=str)
    parser.add_argument('-datapath', type=str)

    args = parser.parse_args()

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]

    net = None
    if args.model == 'ShapeNet32Vox':
        net = model.ShapeNet32Vox()
    elif args.model == 'ShapeNet128Vox':
        net = model.ShapeNet128Vox()
    elif args.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()
    elif args.model == 'SVR':
        net = model.SVR()

    # dataset = voxelized_data.VoxelizedDataset(
    #    args.mode, voxelized_pointcloud=args.pointcloud, pointcloud_samples=args.pc_samples, res=args.res,
    #    sample_distribution=args.sample_distribution, sample_sigmas=args.sample_sigmas,
    #    num_sample_points=100, batch_size=1, num_workers=0)

    num_sample_points = 100
    sample_distribution = np.array(args.sample_distribution)
    sample_sigmas = np.array(args.sample_sigmas)
    num_samples = np.rint(sample_distribution * num_sample_points).astype(np.uint32)
    data_sample = load_data(args.datapath, args.res, args.pc_samples, num_samples, sample_sigmas,
                            voxelized_pointcloud=args.pointcloud)
    # print('Size of inputs {}'.format(data_sample['inputs'].size()))
    # print('Size of grid_coords {}'.format(data_sample['grid_coords'].size()))
    # print('Size of occupancies {}'.format(data_sample['occupancies'].size()))
    # print('Size of points {}'.format(data_sample['points'].size()))

    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format('PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                                    ''.join(str(e) + '_' for e in args.sample_distribution),
                                                    ''.join(str(e) + '_' for e in args.sample_sigmas), args.res,
                                                    args.model)

    gen = Generator(net, 0.5, exp_name, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                    batch_points=args.batch_points)

    mesh = generate_mesh(gen, data_sample)

    mesh.export(os.path.join(SHAPENET_PATH, args.datapath, 'surface_reconstruction.off'))

