import numpy as np
import scipy.ndimage as nd
import scipy.io as io
import skimage.measure as sk
import visdom


def get_voxels_from_mat(path, cube_len=64):
    """Mat to voxel representation"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def get_vertices_and_faces_by_marching_cubes(voxels, threshold=0.5):
    v, f, _, _ = sk.marching_cubes_lewiner(voxels, level=threshold)
    return v, f


def plot_voxels_in_visdom(voxels, vis, plot_title="shape", window_title="shape"):
    """Visdom voxel plot"""
    v, f = get_vertices_and_faces_by_marching_cubes(voxels)
    vis.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=plot_title), win=window_title)


def main():
    with open('../data/ShapeNetChair/chair/train/chair_000002307_4.mat', "rb") as f:
        voxels = np.asarray(get_voxels_from_mat(f, 64), dtype=np.float32)
        vis = visdom.Visdom()
        plot_voxels_in_visdom(voxels, vis, 'chair')


if __name__ == '__main__':
    main()
