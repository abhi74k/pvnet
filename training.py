from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import models
import data

import pvnet_utils

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
        raise NotImplementedError("have not implemented loading from checkpoint")

    model.to(device)
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
        for idx, batch in enumerate(train_data_loader):
            
            preds =  model(batch['img'].to(device))
            net_loss = None
            vector_loss = None
            class_loss = None

            if 'class' in preds.keys():   
                print("calculating class loss...")
                class_target = batch['class_mask']
                print(class_target.size())
                print(torch.reshape(batch['class_label'],(-1,1,1)).size())
                
                # Turn (0,1) mask into labels [0,num_classes] based on class_label
                class_target = class_target * torch.reshape(batch['class_label'],(-1,1,1))
                # Zeros in class mask are actuall "null" class. We want to classify these as index num_classes + 1
                class_target[batch['class_mask'] == 0] = model.num_classes + 1
                print(class_target[0])
                print("Image 1 class label:{}, Min_class: {}, Max_class:".format(batch['class_label'][0],class_target[0].min(), class_target[0].max()))

                # crossentropyloss expects (N, C, H, W) as prediction with probabilities per class c in C, (N, H, W) as output, with values from [0,C-1] for the correct class

                # Because we have single-class labels, we need to ignore non-class losses
                class_loss = class_loss_func(
                    preds['class'], class_target)
                print("Calculated loss: {}".format(class_loss))
                
                # Have per-sample weights

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
        for idx, batch in enumerate(test_data_loader):
            #TODO implement test flow
            print("Testing not implemented")

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

# def _padded_class_masks(class_mask: torch.Tensor
#     , class_label: torch.Tensor
#     , num_classes: int):

#     # Get size of input label, then expand dim=1 to be num_classes+1
#     size = list(class_mask.size())
#     print(size)
#     size[1] = num_classes + 1

#     # Set mask equal to label All other class masks will be zero. 
#     padded_masks = torch.zeros((size))
#     print(padded_masks.size())
#     padded_masks[:,class_label, :,:] = class_mask
    
#     return padded_masks

"""
_padded_vector_label(keypoint_vector_label, class_label)

Dataset only has labels for one class per sample. We need to create padded labels
with zeros for classes other than the one of interest

Take keypoint_vector-label tensor for class_label = i of form (N, 1, H, W)
Return a padded keypoint vector tensor of shape (N, num_classes + 1, H,W)
Where (N, i, H, W) = keypoint_vector-labe
"""
def _padded_vector_label(keypoint_vector_label: torch.Tensor
    , class_Label: torch.Tensor
    , num_classes: int
    , num_keypoints: int):
    
    # Get size of input label, then expand dim=1 to be num_classes+1
    size = list(keypoint_vector_label.size())
    print(size)
    size[1] = num_classes * num_keypoints * 2

    # Set mask equal to label All other class masks will be zero. 
    padded_keypoints = torch.zeros((size))
    # print(padded_keypoints.size())
    padded_keypoints[:,class_Label, :,:] = keypoint_vector_label
    
    raise NotImplementedError()
