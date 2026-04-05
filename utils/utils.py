import cv2

from inference_sdk import InferenceHTTPClient

import numpy as np




def visualize_mask(image, points):
    """
    Utility function to view segmentation mask overlayed on image
    """


    contour = np.array(points).reshape(-1, 1, 2)

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    overlay = image.copy()
    overlay[mask > 0] = (255, 0, 255)

    alpha = 0.5
    vis = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    cv2.imshow("Pipe Segmented Mask", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return vis


def segment_pipe_2d(image_path, api_key):
    """
    Uses Roboflow API to segment pipes
    Returns first pipe mask
    """
    

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    result = client.run_workflow(
        workspace_name="arnav-devalapally",
        workflow_id="general-segmentation-api",
        images={ "image": image_path },
        parameters={ "classes": "pipe" },
        use_cache=True
    )

    points = result[0]['predictions']['predictions'][0]['points']

    polygon = [[int(p['x']), int(p['y'])] for p in points]

    return polygon


def project_segmentation_to_points(point_cloud, intrinsic, extrinsic, polygon):
    """
    Project points onto image, identify points lying inside the segmentation mask
    """

    N = len(point_cloud.points)
    points = np.hstack([point_cloud.points, np.ones((N, 1))])

    camera_matrix = intrinsic.intrinsic_matrix @ extrinsic[:3, :]

    points_image = (camera_matrix @ points.T).T

    u = (points_image[:, 0] / points_image[:, 2]).astype(int)
    v = (points_image[:, 1] / points_image[:, 2]).astype(int)

    mask = np.zeros((intrinsic.height, intrinsic.width))
    contour = np.array(polygon).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 1)

    point_mask = (
        (u >= 0) & (u < intrinsic.width) &
        (v >= 0) & (v< intrinsic.height) &
        (mask[v, u] == 1)
    )

    return point_mask
