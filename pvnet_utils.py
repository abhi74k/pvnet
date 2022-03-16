import random
import os
from importlib import reload

import cv2
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import cv2 as cv
import pandas as pd
import keypoints
import metrics_utils
import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm

import draw_utils

reload(draw_utils)

import torchvision.transforms as T

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

normalize_img = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

denormalize_img = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

img_transforms = T.Compose([
            T.ToTensor(),  # if image.dtype is np.uint8, then it will be divided by 255
            normalize_img
        ])


inverse_img_transforms = T.Compose([
            denormalize_img,
            T.ToPILImage()
        ])


def get_all_labels():
    return list(LABELS.values())


def get_numeric_label(label_str):
    for k, v in LABELS.items():
        if v == label_str:
            return k

    return -1  # not found


def read_pose_file(pose_file_location):
    pose = numpy.load(pose_file_location)
    return pose


def get_files_for_labels(root_dir, labels, shuffle=False):
    results = []
    for label in labels:
        images_path = f'{root_dir}/{label}/JPEGImages/'
        masks_path = f'{root_dir}/{label}/mask/'
        keypoints_path = f'{root_dir}/{label}/labels/'
        pose_path = f'{root_dir}/{label}/pose/'

        images_list = sorted(os.listdir(images_path))
        masks_list = sorted(os.listdir(masks_path))
        keypoints_list = sorted(os.listdir(keypoints_path))
        pose_list = [pose_path + 'pose' + str(pose_i) + ".npy" for pose_i in range(len(images_list))]

        l = [(images_path + image, masks_path + mask, keypoints_path + keypoints, pose, label) for
             image, mask, keypoints, pose in
             zip(images_list, masks_list, keypoints_list, pose_list)]
        random.shuffle(l)
        results.extend(l)

    return np.array(results)


def get_test_train_split(root_dir, labels, test_size=0.33, random_state=42, shuffle=False):
    dataset = get_files_for_labels(root_dir, labels)

    X = dataset[:, 0:4]
    y = dataset[:, 4]

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


def compute_keypoint_vector_pred_error(unit_vectors_preds, unit_vectors_gt, smooth_l1_loss_func, segmentation_gt):
    # ground_truth dimensions : (batch_size, H, W, NUM_KEYPOINTS * 2 * NUM_TRAINING_CLASSES)
    # unit_vectors_pred : ([batch_size,  NUM_KEYPOINTS * 2 * NUM_TRAINING_CLASSES, H, W])
    unit_vectors_preds = unit_vectors_preds.permute(0, 2, 3, 1)

    # [B,H,W] => [B,H,W,1]
    expanded_seg = segmentation_gt.unsqueeze(3)

    # Average only over true pixels in class mask
    # L1 loss function reduces as a sum of all losses. Then divide by # points in class
    loss = smooth_l1_loss_func((unit_vectors_preds * expanded_seg).reshape(-1),
                               (expanded_seg * unit_vectors_gt).reshape(-1)) / expanded_seg.sum()
    return loss


def compute_img_segmentation_pred_error(class_preds, class_mask_gt, class_label_gt, num_classes,
                                        loss_func):
    # cross entropy loss expects (N, C, H, W) as prediction with probabilities per class c in C, (N, H, W) as output,
    # with values from [0,C-1] for the correct class

    # Prediction dim: (batch_size, # class, H, W)
    # Convert ground truth to the same dim as prediction

    # Turn (0,1) mask into labels [0,num_classes] based on class_label
    null_pixels = (class_mask_gt == 0)
    class_mask_gt = class_mask_gt * torch.reshape(class_label_gt, (-1, 1, 1))
    # Zeros in class mask are actual "null" class. We want to classify these as index num_classes + 1
    class_mask_gt[null_pixels] = num_classes
    class_loss = loss_func(class_preds, class_mask_gt)

    return class_loss


def calculate_accuracy(class_preds, class_mask_gt, class_label_gt, num_classes):
    # Save null-class pixels
    null_pixels = class_mask_gt == 0

    # Convert 1D tensor of labels to (N, H, W)
    img_classes_gt = class_mask_gt * torch.reshape(class_label_gt, (-1, 1, 1))

    # Convert 0 to the correct class label
    img_classes_gt[null_pixels] = num_classes

    # (N, C, H, W) -> (N, H, W)
    img_classes_pred = torch.argmax(class_preds, axis=1)

    num_correct_pred = (img_classes_pred == img_classes_gt).sum().item()

    N, H, W = img_classes_pred.size()
    total_entries = N * H * W

    accuracy = num_correct_pred / total_entries

    return accuracy


def load_from_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch

    return start_epoch


def load_model(model, optimizer, model_path, device):
    load_from_checkpoint(model, optimizer, model_path, device)


"""
points3d Nx3 matrix
points2d Nx2 matrix
"""


def solve_pnp(points3d, points2d, method=cv.SOLVEPNP_ITERATIVE):
    assert points3d.shape[0] == points2d.shape[0]

    if method == cv.SOLVEPNP_EPNP:  # Least squares
        points3d = np.reshape(points3d.shape[0], 1, points3d.shape[1])  # (N, 1, 3)
        points2d = np.reshape(points2d.shape[0], 1, points2d.shape[1])  # (N, 1, 2)

    (success, rvec, t) = cv2.solvePnP(np.ascontiguousarray(points3d.astype(np.float64)),
                                      np.ascontiguousarray(points2d.astype(np.float64)),
                                      kinect_camera_matrix,
                                      distCoeffs=np.zeros((4, 1))  # no lens distortion
                                      )

    R = cv.Rodrigues(rvec)  # To convert from angle-axis to rotation matrix

    return rvec, R, t


def get_3d_points(class_label='cat', root_dir=ROOT_DIR):
    points_path = f'{root_dir}/{class_label}/corners.txt'
    print(f'3D points path:{points_path}')
    data = pd.read_csv(points_path, header=None, delimiter=' ')
    return data.to_numpy()


def create_model_and_load_weights(model_weights_path,
                                  device='infer',
                                  num_classes=13,
                                  num_keypoints=9,
                                  output_class=True,
                                  output_vector=True):
    # Create model
    pvnet = models.PvNet(
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        norm_layer=None,
        output_class=output_class,
        output_vector=output_vector)

    optimizer = torch.optim.Adam(pvnet.parameters(), lr=0)

    if device == 'infer':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if device == "cuda" else "cpu")

    load_model(pvnet, optimizer, model_weights_path, device)

    print('Done loading model weights')

    return pvnet

"""
Helper to return x,y,u,v to plot quiver of vector map for a single keypoint 
of a single class
"""
def keypoint_quiver_helper (keypointVector,
                            num_keypoints,
                            segmentation_output,
                            class_idx,
                            keypoint_number=0):
  
  b, c, h, w = keypointVector.size()
  x, y = np.meshgrid(np.linspace(0, w - 1, 50), np.linspace(0, h - 1, 50))

  # Get max clss for pixel, only show u,v for pixels where test_class is most likely class
  _,singleClassMask = torch.max(segmentation_output, dim=0)
  singleClassMask = (singleClassMask==class_idx).unsqueeze(0)
  u, v = keypointVector[0, (class_idx * num_keypoints +keypoint_number)* 2:(class_idx * num_keypoints + keypoint_number + 1) * 2, y,
          x] * singleClassMask.detach()[0:1, y, x]
  v = -v

  return x,y,u,v

def find_keypoints_with_ransac(class_vector_map,
                               test_class,
                               test_class_mask,
                               num_keypoints,
                               ransac_hypotheses,
                               ransac_iterations):
    
    keypointVector = class_vector_map.unsqueeze(0).detach()  # [1, k*2*c,h, w]
    padded_segmentation = test_class_mask.unsqueeze(0)

    x,y,u,v = keypoint_quiver_helper(keypointVector, num_keypoints, test_class_mask, test_class, 0)

    found_keypoints, inClass, hypotheses, vote_cts, vectorPtsInClass = keypoints.findKeypoints(
        padded_segmentation,
        keypointVector,
        [test_class],
        num_hypotheses=ransac_hypotheses,
        max_iterations=ransac_iterations)

    return {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'found_keypoints': found_keypoints
    }


def plot_test_sample(test_sample, class_list, augmented = True):
    img = test_sample['img']
    mask = test_sample['class_mask']
    
    pltimg = inverse_img_transforms(img)
    
    clazz = int(test_sample['class_label'])
    clazz_str = class_list[int(test_sample['class_label'])]

    keypoints = test_sample['obj_keypoints_xy']
    key_x, key_y = zip(*(keypoints.tolist()))

    keypoint_vectors = test_sample['class_vectormap'].permute(2, 0, 1)
    fig = plt.imshow(pltimg)

    fig, axs = plt.subplots(1, 2, figsize=(10, 40 / 3))
    fig.tight_layout()

    # Test image
    axs[0].imshow(pltimg)
    axs[0].scatter(key_x[0:8], key_y[0:8], marker='v', color="red")


    c, h, w = img.size()
    x, y = np.meshgrid(np.linspace(0, w - 1, 50, dtype='int'), np.linspace(0, h - 1, 50, dtype='int'))
    u, v = keypoint_vectors[clazz*keypoints.shape[0]*2:clazz*keypoints.shape[0]*2+2, y, x]
    v = -v  # Sign flip for u,v vs. x,y

    axs[0].quiver(x, y, u, v, color='red', scale=10, scale_units='inches', headwidth=6, headlength=6)
    axs[0].set_title('Label: {0} (#{1})'.format(clazz, clazz_str))

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Label: {0} (#{1})'.format(clazz, clazz_str))

    plt.show()


def plot_nn_segmentation(pred_mask, class_label, class_name):
    plt.figure(figsize=(10, 10))
    im = plt.imshow(pred_mask, cmap='gray')
    plt.title('NN Segmentation for {}({})'.format(class_name, class_label))
    plt.show()

def plot_vector_prediction (img, 
                            vectorPrediction,
                            num_keypoints,
                            segmentation_output,
                            class_idx,
                            obj_keypoints_xy,
                            keypoint_number=0):

    keypointVector = vectorPrediction.unsqueeze(0).to('cpu').detach()  # [1, k*2*c,h, w]   
    x,y,u,v = keypoint_quiver_helper (keypointVector,
                            num_keypoints,
                            segmentation_output.to('cpu'),
                            class_idx,
                            keypoint_number)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(img)
    gt_keypoint = obj_keypoints_xy[keypoint_number]
    plt.scatter(gt_keypoint[0], gt_keypoint[1], marker='v', color="red")
    plt.quiver(x, y, u, v, color='blue', scale=10, scale_units='inches', headwidth=6, headlength=6)    
    plt.title('Vector for {}, Keypoint {}'.format(class_idx, keypoint_number))
    plt.show()

def plot_ransac_results(img, obj_keypoints_xy, 
                          ransac_results, 
                          show_keypoints = True,
                          image_limits = True):
    x, y = ransac_results['x'], ransac_results['y']
    u, v = ransac_results['u'], ransac_results['v']
    found_keypoints = ransac_results['found_keypoints']

    plt.figure(figsize=(10, 10))
    
    plt.imshow(img)
    plt.quiver(x, y, u, v, color='blue', scale=10, scale_units='inches', headwidth=6, headlength=6)
    if show_keypoints:
      plt.scatter(obj_keypoints_xy[:, 0], obj_keypoints_xy[:, 1], marker='v', color="orange", linewidths=5)
      plt.scatter(found_keypoints[:, 0], found_keypoints[:, 1], marker='x', color="blue")
    if image_limits:
      plt.xlim([0, img.size[0]])
      plt.ylim([img.size[1], 0])

    plt.title('RANSAC Keypoint Voting')


def plot_multiclass_mask(class_segmentation, gt_class_label, class_list):
    classviz = torch.max(class_segmentation, dim=0)[1]

    gt_class_name = class_list[gt_class_label]
    num_classes = len(class_list)

    # Normalize to always show from 0 -> num_classes patches
    cm = matplotlib.cm.get_cmap('tab20', num_classes + 1)
    norm = mcolors.Normalize(vmin=0, vmax=num_classes)
    im = plt.imshow(classviz.to('cpu'), norm=norm, cmap=cm, interpolation='none')

    # Create color patches for legend
    colors = [im.cmap(value) for value in range(0, num_classes + 1)]
    patches = [mpatches.Patch(color=colors[i],
                              label="{l}-{i}: {numpix} pixels".format(
                                  l=(class_list[i] if i < len(class_list) else "Null"),
                                  i=i,
                                  numpix=(classviz == i).nonzero().size(0)))
               for i in range(0, num_classes + 1)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title('NN Multi-Class Segmentation (GT Class: {}({}))'.format(gt_class_name, gt_class_label))
    plt.grid(True)
    plt.show()

    '''
K: 3x3 numpy
R: 3x3 numpy
t: 1d vector
points3d: Nx3 dim

Returns:
pixel_coords: Nx2 dim
'''


def project3d_to_2d(K, R, t, points3d):
    camera_coords = (R @ points3d.T) + t.reshape(-1, 1)
    homogenous_image_coords = K @ camera_coords
    pixel_coords = homogenous_image_coords[0:2, :] / homogenous_image_coords[2, :]
    pixel_coords = pixel_coords.T

    return pixel_coords


def make_prediction(pvnet,
                    test_sample,
                    num_keypoints,
                    class_list,
                    root_dir=None,
                    device='cpu',
                    ransac_hypotheses=128,
                    ransac_iterations=10,
                    genplots=True,
                    augmented = True):
    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    pvnet.to(device)

    test_class = int(test_sample['class_label'])
    test_class_str = class_list[int(test_sample['class_label'])]
    
    orig_image = inverse_img_transforms(test_sample['img'])
    
    test_class_mask = test_sample['class_mask'].to(device)
    obj_keypoints_xy = test_sample['obj_keypoints_xy']

    # For each pixel, a vector to each keypoint for each class
    class_vector_map = test_sample['class_vectormap'].to(device)  # [480, 640, k*2*c]

    # Image to tensor
    test_image = test_sample['img'].to(device).unsqueeze(0)

    # Make a prediction
    pred = pvnet(test_image)
    pred_class = pred['class']
    pred_vectors = pred['vector']

    # Load the 3D points for the class
    points3d = get_3d_points(test_class_str, root_dir)

    # Segmentation mask and unit vectors to 2D points using RANSAC

    ransac_results = find_keypoints_with_ransac(pred_vectors[0].to('cpu'),
                                                test_sample['class_label'].to('cpu'),
                                                pred_class[0].to('cpu'),
                                                num_keypoints,
                                                ransac_hypotheses,
                                                ransac_iterations)

    points2d = ransac_results['found_keypoints'][1:9, :].detach().numpy()  # skip 1st point which is the centroid


    # PnP to compute R, t from
    rVec, R, t = solve_pnp(points3d, points2d)

    # Predict image points
    if genplots:
        plot_test_sample(test_sample, class_list, augmented)      
        plot_nn_segmentation(pred_mask=pred_class[0, test_class, :, :].detach().to('cpu').numpy(),
                             class_label=test_class,
                             class_name=class_list[test_class])

        plot_multiclass_mask(pred_class[0], test_class, class_list=class_list)

        plot_vector_prediction(orig_image, 
                                pred_vectors[0], 
                                num_keypoints, 
                                pred_class[0], 
                                test_class,
                                obj_keypoints_xy,
                                1)

        plot_ransac_results(orig_image, 
                          obj_keypoints_xy, 
                          ransac_results, 
                          show_keypoints=True,
                          image_limits=False)      

        image_points_pred = cv2.projectPoints(points3d, rVec, t,
                                              kinect_camera_matrix,
                                              np.zeros(shape=[8, 1], dtype='float64'))[0].squeeze()

        draw_utils.visualize_pose(test_sample, image_points_pred)

    predicted_pose = np.zeros((3, 4))
    predicted_pose[0:3, 0:3] = R[0]
    predicted_pose[0:3, 3] = t.reshape(-1)

    points3d = get_3d_points(test_class_str, root_dir=root_dir)

    return predicted_pose
