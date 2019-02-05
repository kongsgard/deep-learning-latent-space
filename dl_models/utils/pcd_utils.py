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


def create_unit_coordinate_frame():
    return open3d.create_mesh_coordinate_frame(size = 0.5, origin = [0, 0, 0])


class PointsTransform:
    """
    Class to transform points to be axis aligned and centered in the unit sphere, while keeping the inverse transform
    """
    def __init__(self, points):
        self.points = points
        self.max = points.max()
        self.min = points.min()
        self.range = self.max - self.min
        self.mean = points.mean(axis=0)
        self.scaled_mean = 0
        self.has_scaled_to_unit_sphere = False
        self.has_axis_aligned_points = False

    def scale_points_to_unit_sphere(self):
        points_scaled = ((self.points - self.min) / self.range - 0.5) * 2
        self.scaled_mean = points_scaled.mean(axis=0)
        self.points = points_scaled - self.scaled_mean
        self.has_scaled_to_unit_sphere = True

    def axis_align_points(self):
        u, s, vh = np.linalg.svd(self.points, full_matrices=False)
        self.vh = vh
        self.points = np.matmul(self.points, np.transpose(vh))
        self.has_axis_aligned_points = True

    def scale_points_to_original_world_coordinates(self):
        if self.has_axis_aligned_points:
            self.points = np.matmul(self.points, self.vh)
        
        if self.has_scaled_to_unit_sphere:
            points_scaled = self.points + self.scaled_mean
            self.points = (points_scaled/2 + 0.5) * self.range + self.min

    def downscale_points(self, number_of_points):
        pass


if __name__ == '__main__':
    pcd_points = read_pcd('../data/ankylosaurus_mesh.ply')
    # pcd_plane = open3d.read_point_cloud('../data/1a04e3eab45ca15dd86060f189eb133.ply')

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pcd_points)

    points_transformed = PointsTransform(pcd_points)
    
    # Scale points
    points_transformed.scale_points_to_unit_sphere()
    pcd_scaled = open3d.PointCloud()
    pcd_scaled.points = open3d.Vector3dVector(points_transformed.points)

    # Axis-align points
    points_transformed.axis_align_points()
    pcd_aligned = open3d.PointCloud()
    pcd_aligned.points = open3d.Vector3dVector(points_transformed.points)

    # Inverse transform points back to world coordinates
    points_transformed.scale_points_to_original_world_coordinates()
    pcd_world = open3d.PointCloud()
    pcd_world.points = open3d.Vector3dVector(points_transformed.points)

    mesh_frame = create_unit_coordinate_frame()
    custom_draw_geometry_with_key_callback(pcd, pcd_scaled, mesh_frame)
