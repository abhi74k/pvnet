"""
Training for PVNet

"""

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import models
import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

import pvnet_utils


def train(epochs,
          train_data_loader,
          test_data_loader,
          lr=0.0001,
          save="checkpoints/",
          save_suffix = "",
          theta=0.1,
          device="cuda",
          pretrained=False,
          checkpoint_path=None,
          checkpoint_batch_freq=100,
          model=None,
          start_epoch=0):
    device = torch.device("cuda:0" if device == "cuda" else "cpu")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint_path:
        start_epoch = pvnet_utils.load_from_checkpoint(model, optimizer, checkpoint_path, device)
        print("Resuming from: epoch #{}".format(start_epoch + 1))

    num_trainloader = len(train_data_loader)

    # Losses
    class_loss_func = nn.CrossEntropyLoss(
      # ignore_index = model.num_classes    #optional -- ignore "null" class when training 
    ).to(device)
    vector_loss_func = nn.SmoothL1Loss(reduction='sum').to(device)

    # TODO: Implement Tensorboard writing
    log_dir = './runs'
    writer = SummaryWriter(log_dir, comment="pvnet-training", filename_suffix = save_suffix)

    cum_vector_loss = 0
    cum_class_loss = 0

    ## Train
    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()
        running_loss_for_epoch = 0
        accuracy_for_small_batch = 0
        net_loss_for_small_batch = 0
        small_batch_idx = 0

        ## Train
        with tqdm(train_data_loader, unit="batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

                preds = model(batch['img'].to(device))

                vector_loss = torch.zeros(1)
                class_loss = torch.zeros(1)

                if 'class' in preds.keys():
                    class_loss = pvnet_utils.compute_img_segmentation_pred_error(preds['class'],
                                                                                 batch['class_mask'].to(device),
                                                                                 batch['class_label'].to(device),
                                                                                 model.num_classes,
                                                                                 class_loss_func)

                if 'vector' in preds.keys():
                    vector_loss = pvnet_utils.compute_keypoint_vector_pred_error(preds['vector'],
                                                                                 batch['class_vectormap'].to(device),
                                                                                 vector_loss_func,
                                                                                 batch['class_mask'].to(device))

                net_loss = class_loss + vector_loss
                running_loss_for_epoch += net_loss

                accuracy = pvnet_utils.calculate_accuracy(preds['class'], batch['class_mask'].to(device), batch['class_label'].to(device),
                                                          model.num_classes)

                accuracy_for_small_batch += accuracy
                net_loss_for_small_batch += net_loss
                small_batch_idx += 1

                cum_class_loss += class_loss
                cum_vector_loss += vector_loss

                optimizer.zero_grad()
                net_loss.backward()
                optimizer.step()

                print(
                    f"epoch:{epoch + 1}/{epochs}, batch:{idx + 1}/{len(train_data_loader)}, accuracy:{(100.0 * accuracy): .2f}%, "
                    f"total_loss:{net_loss:.4f}, class_loss:{class_loss:.4f}, vector_loss:{vector_loss:.4f}")

                num_iters = epoch * num_trainloader + idx + 1

                if small_batch_idx % 5 == 0:
                    avg_net_loss = net_loss_for_small_batch / small_batch_idx
                    avg_accuracy_for_few_batches = accuracy_for_small_batch / small_batch_idx

                    writer.add_scalar("Train/Total loss", avg_net_loss, num_iters)
                    writer.add_scalar("Train/Vector loss", cum_vector_loss / 5.0, num_iters)
                    writer.add_scalar("Train/Class loss", cum_class_loss / 5.0, num_iters)
                    writer.add_scalar("Train/Accuracy", avg_accuracy_for_few_batches, num_iters)

                    accuracy_for_small_batch = net_loss_for_small_batch = small_batch_idx = 0
                    cum_vector_loss = cum_class_loss = 0

                # Checkpointing logic
                if (idx + 1) % checkpoint_batch_freq == 0:
                    ckpt_path = save + "ckpt_{}{}.pth".format(epoch, save_suffix)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, ckpt_path)

                del preds

        # Final save after end of epoch
        ckpt_path = save + "ckpt_{}_final.pth".format(epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }, ckpt_path)

        print(f'Training loss:{running_loss_for_epoch / len(train_data_loader)}')
