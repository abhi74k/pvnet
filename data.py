import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
import pvnet_utils


class LineModReader(Dataset):

    def __init__(self, dataset, transforms=None):

        self.train_files = dataset[0]
        self.labels = dataset[1]
        self.num_keypoints = 10

        assert self.train_files.shape[0] == len(dataset[1])

        self.dataset_size = self.train_files.shape[0]

        self.imageToTensor = T.ToTensor()
        self.tensorToImage = T.ToPILImage()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        img_path, mask_path, keypoints_path = self.train_files[index, :]

        # Get the keypoints
        class_label, keypoint_coords = pvnet_utils.parse_labels_file(keypoints_path)
        assert class_label == pvnet_utils.get_numeric_label(self.labels[index])

        # Convert (H, W, 3) -> (H, W). Find the coordinates where the image is present
        img_mask = np.array( Image.open(mask_path).convert('1')).astype(np.int32)
        img_mask_coords = np.argwhere(img_mask == 1)

        # Augument each pixel where the image is present with a unit vector pointing to each of the keypoints
        rgb_img = Image.open(img_path)
        img_with_unit_vectors = np.zeros((pvnet_utils.H, pvnet_utils.W, self.num_keypoints * 2))
        pvnet_utils.compute_unit_vectors(img_mask_coords=img_mask_coords, keypoints_coords=keypoint_coords, img_with_unit_vectors=img_with_unit_vectors)

        return self.imageToTensor(rgb_img), torch.tensor(img_mask), img_with_unit_vectors, torch.tensor(class_label).long()

    def show_batch(self, n=3):
        height = 10.0 * n / 3
        fig, axs = plt.subplots(n, 2, figsize=(10, height))
        fig.tight_layout()

        for i in range(n):
            rand_idx = random.randint(0, len(self) - 1)
            img, mask, _, label = self.__getitem__(rand_idx)

            axs[i, 0].imshow(self.tensorToImage(img))
            axs[i, 0].set_title('Label: {0} (#{1})'.format(label.item(), pvnet_utils.LABELS[label.item()]))

            img_mask_scaled = np.where(mask.squeeze().numpy() == 1, [255], [0])
            axs[i, 1].imshow(img_mask_scaled, cmap='gray')
            axs[i, 1].set_title('Label: {0} (#{1})'.format(label.item(), pvnet_utils.LABELS[label.item()]))

