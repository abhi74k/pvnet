import numpy as np
from PIL import Image

from pvnet_utils import get_files_for_labels, ROOT_DIR, parse_labels_file, H, W, compute_unit_vectors
from pvnet_utils import get_test_train_split, get_all_labels


def run_compute_unit_vectors():
    cat_dataset = get_files_for_labels(ROOT_DIR, ['cat'])
    img_path, mask_path, keypoints_path, classname = cat_dataset[0]

    class_label, keypoint_coords = parse_labels_file(keypoints_path)

    img_mask = np.array(Image.open(mask_path).convert('1')).astype(np.int32)
    img_mask_coords = np.argwhere(img_mask == 1)

    img_with_unit_vectors = np.zeros((H, W, 10 * 2))
    compute_unit_vectors(img_mask_coords=img_mask_coords, keypoints_coords=keypoint_coords,
                         img_with_unit_vectors=img_with_unit_vectors)


def run_test_train_split():
    entire_dataset = get_files_for_labels(ROOT_DIR, ['cat'])
    print(entire_dataset.shape)

    X_train, X_test, y_train, y_test = get_test_train_split(ROOT_DIR, get_all_labels()[0:2])

    print(X_train[0:3, :])
    print(y_train[0:3])


if __name__ == '__main__':

    #run_test_train_split()
    run_compute_unit_vectors()