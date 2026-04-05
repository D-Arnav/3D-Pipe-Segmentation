import cv2

import numpy as np

import open3d as o3d

from scipy.spatial.transform import Rotation as R

from utils.utils import visualize_mask, segment_pipe_2d, project_segmentation_to_points
from utils.cylinder import fit_cylinder




config = {
    'data_path': 'dataset/pumps.ply',
    'image_path': 'captures/ScreenCapture_2026-03-31-17-02-21.png',
    'param_path': 'captures/ScreenCamera_2026-03-31-17-02-21.json',
    'roboflow_api_key': ''
}

point_cloud = o3d.io.read_point_cloud(config['data_path'])

image = cv2.imread(config['image_path'])
params = o3d.io.read_pinhole_camera_parameters(config['param_path'])


# 2d segmentation from camera rendering
polygon = segment_pipe_2d(config['image_path'], config['roboflow_api_key'])
visualize_mask(image, polygon)


# project mask to identify points in 3d
pipe_mask = project_segmentation_to_points(point_cloud, params.intrinsic, params.extrinsic, polygon)
pipe_points = point_cloud.select_by_index(np.where(pipe_mask)[0])


# fit cylinder
cylinder = fit_cylinder(np.asarray(pipe_points.points))
(center, axis, radius, height) = cylinder


# visualize 
rot_mat = R.align_vectors([axis], np.array([0, 0, 1]))[0].as_matrix()
cylinder_mesh = o3d.geometry.TriangleMesh \
                   .create_cylinder(radius, height) \
                   .rotate(rot_mat) \
                   .translate(center) \
                   .paint_uniform_color([0.5, 0, 0])


o3d.visualization.draw_geometries([point_cloud, cylinder_mesh])
