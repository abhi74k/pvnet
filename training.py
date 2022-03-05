from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import models
import data

import pvnet_utils


from tensorboard import SummaryWriter

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

    num_trainloader = len(train_data_loader)
    num_testloader = len(test_data_loader)


    # Training utils  
    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Losses
    class_loss_func = nn.CrossEntropyLoss()
    vector_loss_func = nn.SmoothL1Loss()

    # TODO: Implement Tensorboard writing
    log_dir = 'runs'

    ## Train
    for e in range(start_epoch, epochs):
        optimizer.zero_grad() 
        
        ## Train
        for idx, batch in train_data_loader:
            
            preds =  model(batch['img'])
            net_loss = None
            vector_loss = None
            class_loss = None

            if 'class' in preds.keys():               
                class_masks = _padded_class_masks(batch['class_mask'], batch['class_label'], batch.num_classes)
                class_loss = class_loss_func(preds['class'],batch['class_label'])
            else:
                class_loss = torch.zeros(batch_size)

            if 'vector' in preds.keys():
                raise NotImplementedError('Vector classification loss not implemented!')
            else:
               vector_loss = torch.zeros(batch_size)
            
            print("Class loss shape: {}, Vector loss shape: {}:}".format(class_loss.size(),vector_loss.size()))

            net_loss.backward()
            optimizer.step()

            num_iters = e * num_trainloader + idx + 1 
            if (idx + 1 )% 5 == 0 :
                print(
                    "Epoch: #{0} Batch: {1}/{2}\t"
                    # "Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t"
                    # "eta {eta}\t"
                    # "LOSS (current/average) {loss.val:.4f}/{loss.avg:.4f}\t"
                    .format(e+1, idx+1) 
                            # num_trainloader, 
                            # batch_time=batch_time, 
                            # eta=eta, 
                            # loss=net_loss_meter)
                )
                # TODO add tensorbord loss output for class and vector

            # TODO add checkpointing and progress bar outputs

            del preds

        ## Test
        model.eval()
        for idx, batch in test_data_loader:
            batch_size = batch.size(0)

            preds =  model(batch['img'])
            test_net_loss = None
            vector_loss = None
            class_loss = None

            if 'class' in preds.keys():               
                class_masks = _padded_class_masks(batch['class_mask'], batch['class_label'], batch.num_classes)
                class_loss = class_loss_func(preds['class'],batch['class_label'])
            else:
                class_loss = torch.zeros(batch_size)

            if 'vector' in preds.keys():
                raise NotImplementedError('Vector classification loss not implemented!')
            else:
                vector_loss = torch.zeros(batch_size)
            
            print("Class loss shape: {}, Vector loss shape: {}:}".format(class_loss.size(),vector_loss.size()))

            # TODO we need to add metaparameters here around weights?
            test_net_loss = class_loss + vector_loss

            # TODO add tensorbord test loss output


        print(
            "----------------------------------\n"
            "Epoch: #{0}, Avg. Net Test Loss: {test_avg_loss:.4f}\n" 
            "----------------------------------"
            .format(
                epoch+1, test_avg_loss=test_net_loss_meter.avg
            )
        )



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
