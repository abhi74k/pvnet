import numpy as np

import pvnet_utils
from sklearn.neighbors import KDTree
import csv

def compute_add_error(pose_pred, pose_gt, points3d):
    gt_projected = (pose_gt[0:3, 0:3] @ points3d.T + pose_gt[0:3, 3].reshape(-1, 1)).T
    pred_projected = (pose_pred[0:3, 0:3] @ points3d.T + pose_pred[0:3, 3].reshape(-1, 1)).T

    add_metric = np.mean(np.linalg.norm(gt_projected - pred_projected, axis = 1))

    return add_metric


def compute_add_s_error(pose_pred, pose_gt, points3d):
    gt_projected = (pose_gt[0:3, 0:3] @ points3d.T + pose_gt[0:3, 3].reshape(-1, 1)).T
    pred_projected = (pose_pred[0:3, 0:3] @ points3d.T + pose_pred[0:3, 3].reshape(-1, 1)).T

    kdtree = KDTree(pred_projected, metric='euclidean')
    dist, _ = kdtree.query(gt_projected, k=1)

    add_s_metric = np.mean(dist)

    return add_s_metric


def compute_2d_projection_error(pose_pred, pose_gt, points3d):
    gt_2d_projection = pvnet_utils.project3d_to_2d(pvnet_utils.kinect_camera_matrix,
                                                   pose_gt[0:3, 0:3],
                                                   pose_gt[0:3, 3].reshape(-1),
                                                   points3d)

    pred_2d_projection = pvnet_utils.project3d_to_2d(pvnet_utils.kinect_camera_matrix,
                                                     pose_pred[0:3, 0:3],
                                                     pose_pred[0:3, 3].reshape(-1),
                                                     points3d)
    projection_error = np.mean(np.linalg.norm(gt_2d_projection - pred_2d_projection, axis=1))
    return projection_error


def compute_2d_projection_symmetric_error(pose_pred, pose_gt, points3d):
    gt_2d_projection = pvnet_utils.project3d_to_2d(pvnet_utils.kinect_camera_matrix,
                                                   pose_gt[0:3, 0:3],
                                                   pose_gt[0:3, 3].reshape(-1),
                                                   points3d)

    pred_2d_projection = pvnet_utils.project3d_to_2d(pvnet_utils.kinect_camera_matrix,
                                                     pose_pred[0:3, 0:3],
                                                     pose_pred[0:3, 3].reshape(-1),
                                                     points3d)

    kdtree = KDTree(pred_2d_projection, metric='euclidean')
    dist, _ = kdtree.query(gt_2d_projection, k=1)

    projection_error = np.mean(dist)

    return projection_error


def compute_add_error_for_label(pose_pred, pose_gt, points3d, label):
    if label in ['eggbox', 'glue']:
        return compute_add_s_error(pose_pred, pose_gt, points3d)
    else:
        return compute_add_error(pose_pred, pose_gt, points3d)


def compute_2d_projection_error_for_label(pose_pred, pose_gt, points3d, label):
    if label in ['eggbox', 'glue']:
        return compute_2d_projection_symmetric_error(pose_pred, pose_gt, points3d)
    else:
        return compute_2d_projection_error(pose_pred, pose_gt, points3d)


def compute_error_metrics(pose_pred, pose_gt, points3d, label):
    add_error = compute_add_error_for_label(pose_pred, pose_gt, points3d, label)
    projection2d_error = compute_2d_projection_error_for_label(pose_pred, pose_gt, points3d, label)

    return add_error, projection2d_error


"""

returns ADD metric results per batch

Writes metric per sample to file specified at 
'error_metric_dir/error_metrics_{error_file_suffix}.csv'
"""

def compute_error_metrics_for_dataset(test_dataset_reader,
                                      checkpoint_path,
                                      class_list,
                                      root_dir,
                                      error_metric_dir = '',
                                      error_file_suffix = '',                                      
                                      device='cpu',):
    
    points3d = {}
    for label in class_list:
      points3d[label] = pvnet_utils.get_3d_points(label, root_dir=root_dir)

    pvnet = pvnet_utils.create_model_and_load_weights(checkpoint_path, device=device, num_classes=len(class_list))

    add_metric_lst = []
    projection_metric_lst = []

    with open(error_metric_dir+'error_metrics_{}.csv'.format(error_file_suffix), 'w', newline='') as csvfile:
        errorwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        
        errorwriter.writerow(['class_label','pose_path','add_error','projection_error'])
        idx = 0
        for test_sample in test_dataset_reader:
            print("Sample {}".format(idx))

            class_label = test_sample['orig_class_label']
            try:
              assert class_label in class_list

              pred_pose = pvnet_utils.make_prediction(pvnet,
                                                      test_sample,
                                                      pvnet_utils.NUM_KEY_POINTS,
                                                      class_list,
                                                      device=device,
                                                      root_dir=root_dir,
                                                      genplots=False)

              gt_pose = pvnet_utils.read_pose_file(test_sample['pose_path'])

              add_error = compute_add_error_for_label(pred_pose, gt_pose, points3d[label], class_label)
              projection_error = compute_2d_projection_error_for_label(pred_pose, gt_pose, points3d[label], class_label)

              add_metric_lst.append(add_error)
              projection_metric_lst.append(projection_error)
              errorwriter.writerow([class_label,test_sample['pose_path'],add_error,projection_error])

            except Exception as e:
              print("Failed to generate prediction! Error: {}".format(e))
              errorwriter.writerow([class_label,test_sample['pose_path'],"failed to generate prediction"])
            
            idx+=1

    print(f'Avg ADD metric: {np.mean(add_metric_lst)}')
    print(f'Avg 2D projection error: {np.mean(projection_metric_lst)}')

    
