import numpy as np
from PIL import Image
import models
import pvnet_utils
import data
import training
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
import draw_utils

from pvnet_utils import get_files_for_labels, ROOT_DIR, parse_labels_file, H, W, compute_unit_vectors
from pvnet_utils import get_test_train_split, get_all_labels
from pvnet_utils import NUM_TRAINING_CLASSES, NUM_KEY_POINTS

def run_compute_unit_vectors():
    cat_dataset = get_files_for_labels(ROOT_DIR, ['cat'])
    img_path, mask_path, keypoints_path, classname = cat_dataset[0]

    class_label, keypoint_coords = parse_labels_file(keypoints_path)

    img_mask = np.array(Image.open(mask_path).convert('1')).astype(np.int32)
    img_mask_coords = np.argwhere(img_mask == 1)

    img_with_unit_vectors = np.zeros((H, W, NUM_TRAINING_CLASSES * NUM_KEY_POINTS * 2))
    class_relative_offset = int(class_label) * NUM_KEY_POINTS

    compute_unit_vectors(class_offset = class_relative_offset, img_mask_coords=img_mask_coords, keypoints_coords=keypoint_coords,
                         img_with_unit_vectors=img_with_unit_vectors)


def run_test_train_split():
    entire_dataset = get_files_for_labels(ROOT_DIR, ['cat'])
    print(entire_dataset.shape)

    X_train, X_test, y_train, y_test = get_test_train_split(ROOT_DIR, get_all_labels()[0:2])

    print(X_train[0:3, :])
    print(y_train[0:3])


def run_prediction():

    NUM_KEYPOINTS = 2

    X_train, X_test, y_train, y_test = pvnet_utils.get_test_train_split(ROOT_DIR, ['duck', 'cat', 'lamp'],
                                                                        test_size=0.33,
                                                                        random_state=2,
                                                                        shuffle=True)

    test_dataset_reader = data.LineModReader((X_test, y_test), num_keypoints=NUM_KEYPOINTS)
    pvnet = pvnet_utils.create_model_and_load_weights('checkpoints/ckpt_0.pth', device='cpu')
    pvnet_utils.make_prediction(pvnet, test_dataset_reader[0], NUM_KEYPOINTS, root_dir = ROOT_DIR)


def check_pnp():
    NUM_KEYPOINTS = 9
    label = 'cat'

    X_train, X_test, y_train, y_test = get_test_train_split(ROOT_DIR, [label])
    test_dataset_reader = data.LineModReader((X_test, y_test), num_keypoints=NUM_KEYPOINTS)

    pose_path = test_dataset_reader[0]['pose_path']
    pose = np.load(pose_path)

    points3d = pvnet_utils.get_3d_points(label, root_dir=ROOT_DIR)

    R = pose[:, 0:3]
    t = pose[:, 3]
    K = pvnet_utils.kinect_camera_matrix

    uv = pvnet_utils.project3d_to_2d(K, R, t, points3d)
    rVec, R, t = pvnet_utils.solve_pnp(points3d, uv)

    image_points_pred = cv2.projectPoints(points3d, rVec, t,
                                          pvnet_utils.kinect_camera_matrix,
                                          np.zeros(shape=[8, 1], dtype='float64'))[0].squeeze()

    draw_utils.visualize_pose(test_dataset_reader[0], image_points_pred)

if __name__ == '__main__':

    #run_test_train_split()
    #run_compute_unit_vectors()
    #run_prediction()
    check_pnp()
