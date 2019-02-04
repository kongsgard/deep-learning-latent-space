import matplotlib.pyplot as plt
import numpy as np
import open3d


def read_pcd(filename):
    pcd = open3d.read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    open3d.write_point_cloud(filename, pcd)


def custom_draw_geometry_with_key_callback(pcd, *args):
    def load_render_option(vis):
        vis.get_render_option().load_from_json("assets/renderoption.json")
        return False
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    key_to_callback = {}
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    open3d.draw_geometries_with_key_callbacks([pcd, *args], key_to_callback)


def scale_points_to_unit_sphere(points):
    max_point = points.max()
    min_point = points.min()

    points_range = max_point - min_point
    points_scaled = ((points - min_point) / points_range - 0.5) * 2

    return points_scaled - points_scaled.mean(axis=0)

if __name__ == '__main__':
    pcd_points = read_pcd('../data/ankylosaurus_mesh.ply')
    pcd_plane = open3d.read_point_cloud('../data/1a04e3eab45ca15dd86060f189eb133.ply')

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pcd_points)

    pcd_scaled = open3d.PointCloud()
    pcd_scaled.points = open3d.Vector3dVector(scale_points_to_unit_sphere(pcd_points))

    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.5, origin = [0, 0, 0])
    custom_draw_geometry_with_key_callback(pcd, pcd_scaled, pcd_plane, mesh_frame)
    