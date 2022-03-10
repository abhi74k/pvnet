import random
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import cv2 as cv

ROOT_DIR = "dataset/LINEMOD"

LABELS = {
    0: 'ape',
    1: 'benchvise',
    2: 'cam',
    3: 'can',
    4: 'cat',
    5: 'driller',
    6: 'duck',
    7: 'eggbox',
    8: 'glue',
    9: 'holepuncher',
    10: 'iron',
    11: 'lamp',
    12: 'phone'
}

H = 480
W = 640

NUM_KEY_POINTS = 9  # 1 centroid + 8 bounding box corners
NUM_CLASSES = len(list(LABELS.values()))
NUM_TRAINING_CLASSES = NUM_CLASSES + 1  # To indicate none of the objects were found

kinect_camera_matrix = np.array([
    [572.4114, 0., 325.2611],
    [0., 573.57043, 242.04899],
    [0., 0., 1.]])


def get_all_labels():
    return list(LABELS.values())


def get_numeric_label(label_str):
    for k, v in LABELS.items():
        if v == label_str:
            return k

    return -1  # not found


def get_files_for_labels(root_dir, labels, shuffle=False):
    results = []
    for label in labels:
        images_path = f'{root_dir}/{label}/JPEGImages/'
        masks_path = f'{root_dir}/{label}/mask/'
        keypoints_path = f'{root_dir}/{label}/labels/'

        images_list = sorted(os.listdir(images_path))
        masks_list = sorted(os.listdir(masks_path))
        keypoints_list = sorted(os.listdir(keypoints_path))

        l = [(images_path + image, masks_path + mask, keypoints_path + keypoints, label) for image, mask, keypoints in
             zip(images_list, masks_list, keypoints_list)]
        random.shuffle(l)
        results.extend(l)

    return np.array(results)


def get_test_train_split(root_dir, labels, test_size=0.33, random_state=42, shuffle=False):
    dataset = get_files_for_labels(root_dir, labels)

    X = dataset[:, 0:3]
    y = dataset[:, 3]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle)

    return X_train, X_test, y_train, y_test


def parse_labels_file(keypoints_path, num_keypoints=None):
    with open(keypoints_path) as f:
        keypoints_str = f.readline()

    keypoints_str_lst = keypoints_str.split(' ')
    class_label = int(keypoints_str_lst[0])
    keypoints_coords = np.array([float(x) for x in keypoints_str_lst[1:]]).reshape(-1, 2)

    # Datafile may have more keypoints than we want. If specified, only take num_keypoints
    if num_keypoints is not None:
        keypoints_coords = keypoints_coords[0:num_keypoints, :]

    return class_label, keypoints_coords


def compute_unit_vectors(class_offset, img_mask_coords, keypoints_coords, img_with_unit_vectors):
    keypoints_xy_coords = keypoints_coords * [W, H]  # x, y
    img_mask_xy_coords = img_mask_coords[:, [1, 0]]

    nrows = keypoints_xy_coords.shape[0]
    for keypoint_idx in np.arange(nrows):
        keypoint_xy = keypoints_xy_coords[keypoint_idx]
        keypoint_dir_vector_xy = keypoint_xy - img_mask_xy_coords
        keypoint_dir_vector_mag = np.linalg.norm(keypoint_dir_vector_xy, axis=1).reshape(-1, 1)
        keypoint_dir_unit_vector_xy = keypoint_dir_vector_xy / keypoint_dir_vector_mag

        img_mask_x_coords = img_mask_xy_coords[:, 0]
        img_mask_y_coords = img_mask_xy_coords[:, 1]

        img_with_unit_vectors[
            img_mask_y_coords, img_mask_x_coords, class_offset + 2 * keypoint_idx] = keypoint_dir_unit_vector_xy[:, 0]
        img_with_unit_vectors[
            img_mask_y_coords, img_mask_x_coords, class_offset + 2 * keypoint_idx + 1] = keypoint_dir_unit_vector_xy[:,
                                                                                         1]


def compute_keypoint_vector_pred_error(unit_vectors_preds, unit_vectors_gt, smooth_l1_loss_func):
    # ground_truth dimensions : (batch_size, H, W, NUM_KEYPOINTS * 2 * NUM_TRAINING_CLASSES)
    # unit_vectors_pred : ([batch_size,  NUM_KEYPOINTS * 2 * NUM_TRAINING_CLASSES, H, W])
    unit_vectors_preds = unit_vectors_preds.permute(0, 2, 3, 1)
    loss = smooth_l1_loss_func(unit_vectors_preds.reshape(-1), unit_vectors_gt.reshape(-1))
    return loss


def compute_img_segmentation_pred_error(class_preds, class_mask_gt, class_label_gt, num_classes,
                                        cross_entropy_loss_func):
    # cross entropy loss expects (N, C, H, W) as prediction with probabilities per class c in C, (N, H, W) as output,
    # with values from [0,C-1] for the correct class

    # Prediction dim: (batch_size, # class, H, W)
    # Convert ground truth to the same dim as prediction

    # Turn (0,1) mask into labels [0,num_classes] based on class_label
    class_mask_gt = class_mask_gt * torch.reshape(class_label_gt, (-1, 1, 1))
    # Zeros in class mask are actual "null" class. We want to classify these as index num_classes + 1
    class_mask_gt[class_mask_gt == 0] = num_classes

    # Because we have single-class labels, we need to ignore non-class losses
    class_loss = cross_entropy_loss_func(class_preds, class_mask_gt)

    return class_loss


def calculate_accuracy(class_preds, class_mask_gt, class_label_gt, num_classes):
    # Convert 1D tensor of labels to (N, H, W)
    img_classes_gt = class_mask_gt * torch.reshape(class_label_gt, (-1, 1, 1))

    # Convert 0 to the correct class label
    img_classes_gt[img_classes_gt == 0] = num_classes

    # (N, C, H, W) -> (N, H, W)
    img_classes_pred = torch.argmax(class_preds.cpu(), axis=1)

    num_correct_pred = (img_classes_pred == img_classes_gt).sum().item()

    N, H, W = img_classes_pred.size()
    total_entries = N * H * W

    accuracy = num_correct_pred / total_entries

    return accuracy


def load_from_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch

    return start_epoch


def load_model(model, optimizer, model_path, device):
    load_from_checkpoint(model, optimizer, model_path, device)


"""
points3d Nx3 matrix
points2d Nx2 matrix
"""


def solve_pnp(points3d, points2d, camera_matrix, method=cv.SOLVEPNP_ITERATIVE):
    assert points3d.shape[0] == points2d.shape[0]

    if method == cv.SOLVEPNP_EPNP:  # Least squares
        points3d = np.reshape(points3d.shape[0], 1, points3d.shape[1])  # (N, 1, 3)
        points2d = np.reshape(points2d.shape[0], 1, points2d.shape[1])  # (N, 1, 2)

    (success, rvec, t) = cv2.solvePnP(np.ascontiguousarray(points3d.astype(np.float64)),
                                      np.ascontiguousarray(points2d.astype(np.float64)),
                                      camera_matrix,
                                      distCoeffs=np.zeros((4, 1))  # no lens distortion
                                      )

    R = cv.Rodrigues(rvec)  # To convert from angle-axis to rotation matrix

    return rvec, R, t
