import numpy as np

import pvnet_utils
from sklearn.neighbors import KDTree

def compute_add_metric(pose_pred, pose_gt, points3d):
    gt_projected = (pose_gt[0:3, 0:3] @ points3d.T + pose_gt[0:3, 3].reshape(-1, 1)).T
    pred_projected = (pose_pred[0:3, 0:3] @ points3d.T + pose_pred[0:3, 3].reshape(-1, 1)).T

    add_metric = np.mean(np.linalg.norm(gt_projected - pred_projected))

    return add_metric


def compute_add_s_metric(pose_pred, pose_gt, points3d):
    gt_projected = (pose_gt[0:3, 0:3] @ points3d.T + pose_gt[0:3, 3].reshape(-1, 1)).T
    pred_projected = (pose_pred[0:3, 0:3] @ points3d.T + pose_pred[0:3, 3].reshape(-1, 1)).T

    kdtree = KDTree(pred_projected, metric='euclidean')
    dist, _ = kdtree.query(gt_projected, k=1)

    add_s_metric = np.mean(dist)

    return add_s_metric


def compute_add_metric_for_label(test_dataset_reader,
                                 checkpoint_path,
                                 label,
                                 device='cpu',
                                 root_dir=pvnet_utils.ROOT_DIR):
    points3d = pvnet_utils.get_3d_points(label, root_dir=root_dir)

    pvnet = pvnet_utils.create_model_and_load_weights(checkpoint_path, device=device, num_classes=1)

    add_metric_lst = []

    for test_sample in test_dataset_reader:

        assert test_sample['orig_class_label'] == label

        pred_pose = pvnet_utils.make_prediction(pvnet,
                                                test_sample,
                                                pvnet_utils.NUM_KEY_POINTS,
                                                [label],
                                                device=device,
                                                root_dir=root_dir,
                                                genplots=False)

        gt_pose = pvnet_utils.read_pose_file(test_sample['pose_path'])

        if label in ['eggbox', 'glue']:
            add_metric_lst.append(compute_add_s_metric(pred_pose, gt_pose, points3d))
        else:
            add_metric_lst.append(compute_add_s_metric(pred_pose, gt_pose, points3d))

    print(f'Avg ADD metric: {np.mean(add_metric_lst)}')
