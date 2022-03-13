import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
import pvnet_utils
from math import cos, sin, radians
from pvnet_utils import NUM_CLASSES, NUM_KEY_POINTS, LABELS







class LineModReader(Dataset):

    def __init__(self,
                  dataset, 
                  class_list, 
                  augmentation=False,
                  num_keypoints=NUM_KEY_POINTS):
        self.train_files = dataset[0]
        self.labels = dataset[1]
        self.num_keypoints = num_keypoints
        
        self.class_list = class_list
        self.num_classes = len(class_list)

        assert self.train_files.shape[0] == len(dataset[1])

        self.dataset_size = self.train_files.shape[0]

        self.imageToTensor = T.ToTensor()
        self.tensorToImage = T.ToPILImage()

        self.augmentation = augmentation

    def __len__(self):
        return self.dataset_size

    # Returns img_tensor, segmentation_mask, unit_vector, class_label, keypoints
    def __getitem__(self, index):
        img_path, mask_path, keypoints_path, pose_path = self.train_files[index, :]

        # Get the keypoints
        class_label_str, keypoint_coords = pvnet_utils.parse_labels_file(keypoints_path, self.num_keypoints)
        assert class_label_str == pvnet_utils.get_numeric_label(self.labels[index])

        # We cannot assume that class_idx in the dataset will be the same as hardcoded labels
        class_name = LABELS[int(class_label_str)]
        class_idx = self.class_list.index(class_name)

        # Keypoints are % of image W and Height
        keypoint_xy_coords = keypoint_coords * [pvnet_utils.W, pvnet_utils.H]

        # Convert (H, W, 3) -> (H, W). Find the coordinates where the image is present
        img_mask = np.array(Image.open(mask_path).convert('1')).astype(np.int32)
        img_mask_coords = np.argwhere(img_mask == 1)

        # Augument each pixel where the image is present with a unit vector pointing to each of the keypoints
        rgb_img = Image.open(img_path)
        img_with_unit_vectors = np.zeros((pvnet_utils.H, pvnet_utils.W, self.num_classes * self.num_keypoints * 2))
        class_relative_offset = class_idx * self.num_keypoints *2
        pvnet_utils.compute_unit_vectors(class_relative_offset, img_mask_coords=img_mask_coords,
                                         keypoints_coords=keypoint_coords, img_with_unit_vectors=img_with_unit_vectors)

        sample = {
            'img': self.imageToTensor(rgb_img),   #[3,h,w]
            'class_mask': torch.tensor(img_mask).type(torch.LongTensor),  #[h,w]
            'class_vectormap': torch.tensor(img_with_unit_vectors).type(torch.FloatTensor), #[h,w,num_c*num_k*2]
            'class_label': torch.tensor(class_idx).long(),
            'obj_keypoints_xy': keypoint_xy_coords,
            'obj_keypoints': keypoint_coords,
            'pose_path': pose_path
        }

        # Apply transforms if provided
        if self.augmentation:
          self.applyTransforms(sample)

        return sample
    
    """
    Regular random transforms can't be applied to each label, because random
    values will differ and be misaligned. We need to stack them into
    one tensor, apply transforms at once, and then unpack tensor.
    
    """
    def applyTransforms(self, sample):
      C,H,W = sample['img'].size()

      # Apply color jitter only to rgb image
      sample['img'] = T.ColorJitter()(sample['img'])

      # Reshape and join into one multi-channel tensor
      stacked = torch.cat((sample['img'],                       #[3,H,W]
                            sample['class_mask'].unsqueeze(0),  #[H,W]=>[1,H,W]
                            sample['class_vectormap'].permute(2,0,1)  #[H,W,C]=>[C,H,W]
                            ), dim = 0)

      
      # Transforms applied to all matrices equally (only RandomResizedCrop for now)
      joint_transforms = T.Compose([T.RandomResizedCrop((H,W))])
      stacked = joint_transforms(stacked)

      # Apply rotation. Need to save angle to 
      angle = float(torch.empty(1).uniform_(0, 360).item())
      rads = radians(angle)

      # Transform between uv and xy before applying rotation
      uv_to_xy = torch.Tensor([1,0,0,-1]).reshape(2,2)
      R = torch.Tensor([[cos(rads), -sin(rads)]
                        ,[sin(rads), cos(rads)]]).type(torch.FloatTensor)

      stacked = T.functional.rotate(stacked, angle, fill=0)

      # Unpack original labels
      sample['img'] = stacked[0:3]
      sample['class_mask'] = stacked[3].type(torch.LongTensor)

      # Rotate vector values
      vector_temp = stacked[4:].permute(1,2,0)
      H, W, C = vector_temp.size()
      vector_temp = vector_temp.reshape(H,W,round(C/2),2) #[H,W,numpts,(xy)]
      vector_temp = vector_temp.permute(0,1,3,2) #[H,W,(xy),numpts]
      vector_temp = uv_to_xy @ R @ uv_to_xy @ vector_temp.type(torch.FloatTensor)
      sample['class_vectormap'] = vector_temp.permute(0,1,3,2).reshape(H,W,C)


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
            keypoint_vectors = sample['class_vectormap'].permute(2, 0, 1)

            key_x, key_y = zip(*(keypoints.tolist()))
            axs[i, 0].imshow(self.tensorToImage(img))
            axs[i, 0].scatter(key_x[0:8], key_y[0:8], marker='v', color="red")

            # viz vector field arrows for centroid keypoint
            c, h, w = img.size()
            x, y = np.meshgrid(np.linspace(0, w - 1, 50, dtype='int'), np.linspace(0, h - 1, 50, dtype='int'))
            u, v = keypoint_vectors[0:2, y, x]

            # Sign flip for u,v vs. x,y
            v = -v

            axs[i, 0].quiver(x, y, u, v, color='red', scale=10, scale_units='inches', headwidth=6, headlength=6)

            axs[i, 0].set_title('Label: {0} (#{1})'.format(label.item(), self.class_list[label.item()]))

            img_mask_scaled = np.where(mask.squeeze().numpy() == 1, [255], [0])
            axs[i, 1].imshow(img_mask_scaled, cmap='gray')
            axs[i, 1].set_title('Label: {0} (#{1})'.format(label.item(), self.class_list[label.item()]))
