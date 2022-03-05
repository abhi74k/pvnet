from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import models
import data

from tensorboard import SummaryWriter
from utils import AverageMeter, colorize, init_training_vals

def train(epochs, 
        train_data_loader,
        test_data_loader,
        lr=0.0001, 
        save="checkpoints/", 
        theta=0.1, 
        device="cuda", 
        pretrained=False,
        checkpoint=None,
        model=None,
        start_epoch=0):

    # TODO: Implement start from checkpoint
    if checkpoint != None:
        raise NotImplementedError("have not implemented loading from checkpoint"i)

    model.to(device)
    test_data_loader.to(device)
    train_data_loader.to(device)
    model.train()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Losses
    class_loss_func = nn.CrossEntropyLoss()
    vector_loss_func = nn.SmoothL1Loss()

    # TODO: Implement Tensorboard writing

    ## Train
    for e in range(start_epoch, epochs):

        optimizer.zero_grad() 
        for idx, batch in train_data_loader:
            
            preds =  model(batch['img'])

            net_loss = None
            vector_loss = None
            class_loss = None

            if 'class' in preds.keys():               
                class_masks = _padded_class_masks(batch['class_mask'], batch['class_label'], batch.num_classes)
                print(class_masks.size())
                # class_loss = class_loss_func(preds['class'],class_label)

            if 'vector' in preds.keys():
                raise NotImplementedError('Vector classification loss not implemented!')


"""
_padded_class_label(keypoint_vector_label, class_label)

Dataset only has labels for one class per sample. We need to create padded labels
with zeros for classes other than the one of interest

Take keypoint_vector-label tensor for class_label = i of form (N, 1, H, W)
Return a padded keypoint vector tensor of shape (N, num_classes + 1, H,W)
Where (N, i, H, W) = keypoint_vector-label

"""

def _padded_class_masks(class_mask: torch.Tensor
    , class_label: torch.Tensor
    , num_classes: int):

    # Get size of input label, then expand dim=1 to be num_classes+1
    size = keypoint_vector_label.size()
    size[1] = num_classes + 1

    # Set mask equal to label All other class masks will be zero. 
    padded_masks = torch.zeros((size))
    padded_masks[:,class_label, :,:] = class_mask
    
    return padded_masks

"""
_padded_vector_label(keypoint_vector_label, class_label)
Take keypoint_vector-label tensor for class_label = i of form (N, num_keypoints * 2, H, W)
Return a padded keypoint vector tensor of shape (N, num_classes * num_keypoints * 2, H,W)
Where (N, i, H, W) = keypoint_vector-label
"""
def _padded_vector_label(keypoint_vector_label: torch.Tensor
    , class_Label: torch.Tensor
    , num_classes: int):
    raise NotImplementedError()
