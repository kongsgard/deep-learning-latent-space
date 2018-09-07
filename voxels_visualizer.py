import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.ndimage as nd
import scipy.io as io
import skimage.measure as sk
import visdom

from plotly.offline import plot
import plotly.graph_objs as go

def get_voxels_from_mat(path, cube_len=64):
    """Mat to voxel representation"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def plot_voxels_in_matplotlib(voxels):
    """Matplotlib voxel plot - VERY slow"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='red', edgecolor='white', linewidth=0.1)
    plt.show()


def plot_voxels_in_plotly(voxels):
    """Plotly scatter plot"""
    z, x, y = voxels.nonzero()
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=12,
            color='#FFB6C1',
            opacity=0.8
        )
    )
    plot([trace])


def get_vertices_and_faces_by_marching_cubes(voxels, threshold=0.5):
    v, f, _, _ = sk.marching_cubes_lewiner(voxels, level=threshold)
    return v, f


def plot_voxels_in_visdom(voxels, visdom, title):
    """Visdom voxel plot - fast"""
    v, f = get_vertices_and_faces_by_marching_cubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title), win='chair')


def main():
    with open('data/ShapeNetChair/chair/train/chair_000002307_4.mat', "rb") as f:
        voxels = np.asarray(get_voxels_from_mat(f, 64), dtype=np.float32)
        #plot_voxels_in_matplotlib(voxels)
        #plot_voxels_in_plotly(voxels)
        vis = visdom.Visdom()
        plot_voxels_in_visdom(voxels, vis, 'chair')


if __name__ == '__main__':
    main()