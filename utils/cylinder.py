import numpy as np




def fit_cylinder(point_cloud):
    """
    Fits cylinder by
    - Estimating center and axis via mean and SVD
    - Finding mean squared estimate from radius
    """

    center = point_cloud.mean(0)
    point_cloud_centered = point_cloud - center
    
    Vt = np.linalg.svd(point_cloud_centered, full_matrices=False)[2]
    axis = Vt[0] / np.linalg.norm(Vt[0])

    point_cloud_axis_proj = point_cloud_centered @ axis

    height = point_cloud_axis_proj.max() - point_cloud_axis_proj.min()
    center = center + axis * (point_cloud_axis_proj.max() + point_cloud_axis_proj.min()) / 2
    radius = ((point_cloud_centered - point_cloud_axis_proj[:, None] * axis) ** 2).sum(1).mean() ** 0.5

    return (center, axis, radius, height)

