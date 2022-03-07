import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
import pvnet_utils


class LineModReader(Dataset):

    def __init__(self, dataset, transforms=None, num_keypoints = 9):

        self.train_files = dataset[0]
        self.labels = dataset[1]
        self.num_keypoints = num_keypoints

        assert self.train_files.shape[0] == len(dataset[1])

        self.dataset_size = self.train_files.shape[0]

        self.imageToTensor = T.ToTensor()
        self.tensorToImage = T.ToPILImage()

    def __len__(self):
        return self.dataset_size


    # Returns img_tensor, segmentation_mask, unit_vector, class_label, keypoints
    def __getitem__(self, index):

        img_path, mask_path, keypoints_path = self.train_files[index, :]

        # Get the keypoints
        class_label, keypoint_coords = pvnet_utils.parse_labels_file(keypoints_path, self.num_keypoints)
        assert class_label == pvnet_utils.get_numeric_label(self.labels[index])
        
        # Keypoints are % of image W and Height
        keypoint_xy_coords = keypoint_coords*[pvnet_utils.W,pvnet_utils.H]

        # Convert (H, W, 3) -> (H, W). Find the coordinates where the image is present
        img_mask = np.array( Image.open(mask_path).convert('1')).astype(np.int32)
        img_mask_coords = np.argwhere(img_mask == 1)

        # Augument each pixel where the image is present with a unit vector pointing to each of the keypoints
        rgb_img = Image.open(img_path)
        img_with_unit_vectors = np.zeros((pvnet_utils.H, pvnet_utils.W, self.num_keypoints * 2))
        pvnet_utils.compute_unit_vectors(img_mask_coords=img_mask_coords, keypoints_coords=keypoint_coords, img_with_unit_vectors=img_with_unit_vectors)

        sample = {
            'img': self.imageToTensor(rgb_img), 
            'class_mask': torch.tensor(img_mask).type(torch.LongTensor),
            'class_vectormap': img_with_unit_vectors, 
            'class_label': torch.tensor(class_label).long(),
            'obj_keypoints_xy': keypoint_xy_coords,
            'obj_keypoints': keypoint_coords
        }

        return sample

    def show_batch(self, n=3):
        height = 10.0 * n / 3
        fig, axs = plt.subplots(n, 2, figsize=(10, height))
        fig.tight_layout()

        for i in range(n):
            rand_idx = random.randint(0, len(self) - 1)
            sample = self.__getitem__(rand_idx)
            img = sample['img']
            mask = sample['class_mask']
            label = sample['class_label']
            keypoints = sample['obj_keypoints'] 
            keypoint_vectors = sample['class_vectormap'].transpose(2,0,1)
            
            key_x, key_y = zip(*(keypoints.tolist()))
            axs[i, 0].imshow(self.tensorToImage(img))
            axs[i, 0].scatter(key_x[0:8],key_y[0:8], marker='v', color="red")
            
            # viz vector field arrows for centroid keypoint
            c, h, w = img.size()
            x,y = np.meshgrid(np.linspace(0,w-1,50, dtype='int'),np.linspace(0,h-1,50,dtype='int'))
            u,v = keypoint_vectors[0:2,y,x]

            # Sign flip for u,v vs. x,y
            v = -v

            axs[i,0].quiver(x,y,u,v, color= 'red', scale = 10, scale_units = 'inches', headwidth=6, headlength=6)

            axs[i, 0].set_title('Label: {0} (#{1})'.format(label.item(), pvnet_utils.LABELS[label.item()]))

            img_mask_scaled = np.where(mask.squeeze().numpy() == 1, [255], [0])
            axs[i, 1].imshow(img_mask_scaled, cmap='gray')
            axs[i, 1].set_title('Label: {0} (#{1})'.format(label.item(), pvnet_utils.LABELS[label.item()]))
