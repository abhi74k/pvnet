import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = "../dataset/LINEMOD"

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


def parse_labels_file(keypoints_path, num_keypoints = None):
    with open(keypoints_path) as f:
        keypoints_str = f.readline()

    keypoints_str_lst =  keypoints_str.split(' ')
    class_label = int(keypoints_str_lst[0])
    keypoints_coords = np.array([float(x) for x in keypoints_str_lst[1:]]).reshape(-1, 2)

    # Datafile may have more keypoints than we want. If specified, only take num_keypoints
    if num_keypoints is not None:
        keypoints_coords = keypoints_coords[0:num_keypoints, :]


    return class_label, keypoints_coords


def compute_unit_vectors(img_mask_coords, keypoints_coords, img_with_unit_vectors):

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

        img_with_unit_vectors[img_mask_y_coords, img_mask_x_coords, 2 * keypoint_idx] = keypoint_dir_unit_vector_xy[:, 0]
        img_with_unit_vectors[img_mask_y_coords, img_mask_x_coords, 2 * keypoint_idx + 1] = keypoint_dir_unit_vector_xy[:, 1]

