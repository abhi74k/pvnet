from typing import Sequence
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image

import resnet


class UpsamplingBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        input: int,
        output: int,
        scale_factor: Optional[nn.Module] = None,
        upsample_mode: str = "bilinear",
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(UpsamplingBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = resnet.conv3x3(input, output, stride = 1)
        self.bn1 = norm_layer(output)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(output, scale_factor, upsample_mode)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.upsample(out)
        return out


class PvNet(resnet.ResNet):
    """
    Given C = num_classes, K = num_keypoints per class
    
    Take input RGB image, if 'output_class' then output semantic seg ((C+1)xHxW), if 'output_vector' then
    output pixel-wise keypoint vector ((C*K+1)xHxW). 18 because 2 channels for vector (u,v) pointing to each
    keypoint, for 9 keypoints

    The first half is a modified version of ResNet-18, but modifies layer4 and the final pooling/fc step
    so that the image dimensionality is not reduced below [H/8,W/8]

    We still try to initialize the first few layers with a pre-trained set of weights from canonical ResNet-18

    TODO: Consider separating into two separate models -- the ResNet downsampling, then the upsampling
    """

    def __init__(self,
        num_classes: int = 12,
        num_keypoints: int = 9,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        output_class = True,
        output_vector = False
        ):
       
        # TODO: Implement for vector outputs too then remove
        if output_vector:
            raise NotImplementedError

        self.output_class = output_class
        self.output_vector = output_vector

        # Init from grandparent, not ResNet function because we have funky stuff to do
        super().super().__init__()
        
        # Hardcode BasicBlock because we are only using ResNet18, which doesn't use BottleneckBlock
        block = resnet.BasicBlock
        # Hardcode layer block count as all 2,s because we are doing ResNet 18
        layers= [2,2,2,2]

        # Below is mostly copied from ResNet.__init__() but with important modifications
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, True, True]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        ## Make 4 "Residual" layers with passthrough of residual
        # _make_layer() is inherited as-is from ResNet implementation
        
        # First layer has no stride
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Second layer downsample
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=False)
        
        # Third layer downsample again
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=False)
        
        # Unlike ResNet 18, for layer 4 we don't striding by 2 so we stop downsampling image
        # We dilate by 2x so the kernel still "covers" the same range of pixels it would stride = 2
        self.dilation *= 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=True)
       
        ## Change from ResNet -- there is no pool or fc layer at the end. Replace with Convolutional layer
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # TODO: Check with Chen if this is the right convolution
        self.reverseConv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=7, stride=1, padding=3,
                                bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True)                               
        )

        ## Add upsampling layers
        
        self.upsample1 = UpsamplingBlock(256, 128, 2, "bilinear")
        self.upsample2 = UpsamplingBlock(128, 64, 2, "bilinear")
        self.upsample3 = UpsamplingBlock(256, 32, 2, "bilinear")

        # Final convolution before output layers
        self.endConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                                bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True)                               
        )

        
        ## Output convolutions
        # 1x1 convolutions over full depth, reshaping to appropriate output

        # For class, output num_classes+1 channels with probability for each class (including one null class)
        if self.output_class:
            self.class_out = nn.Conv2D(32, num_classes + 1, kernel_size=1, stride=1, bias=False)

        # For vector, output num_classes*num_keypoints*2, since each class can have num_keypoints, and each keypoint has [u,v]
        if self.output_vector:
            self.vector_out = nn.Conv2D(3, num_keypoints * num_classes * 2, kernel_size=1, stride=1, bias=False)
        
        ## TODO: Initialize all modules above with ResNet18 pre-trained weights, where they exist
        #
        #  DO THAT HERE
        #



    # This mostly matches ResNet._forward_impl but retains skip residual values for use in upsampling steps
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip1 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        skip2 = x
        x = self.layer2(x)
        skip3 = x
        x = self.layer3(x)
        skip4 = x
        x = self.layer4(x)

        # We don't use these steps in PvNet
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        
        x = self.endConv(x)

        # First skip layer addback
        x = self.upsample1(x + skip4)
        x = self.upsample2(x + skip3)
        x = self.upsample3(x + skip2)
        x = self.endConv(x + skip1)
        
        outputs = []

        if self.output_class:
            outputs+= [self.class_out(x)]

        if self.output_vector:
            outputs+= [self.vector_out(x)]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)




if __name__ == "__main__":
    resnet18 = models.resnet18(pretrained=True)
    newmodel = PvNet()
    # print(resnet18)
    i = 0
    for name, mod in resnet18.named_children():
        if i>4: break
        newmodel.add_module(name, mod)
        i+=1
    # print(newmodel)
    print(resnet18)

    # for name, child in newmodel.named_children():
    #     print("name: {}, child_params: {}".format(name,list(child.named_parameters())))


    
    # print(resnet18.get_submodule("layer4"))
    # for param in resnet18.parameters():
        # print(name)
        # print(type(param),param.size())
        # print(param)
    pass
