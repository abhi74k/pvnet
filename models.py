import torch
import torch.nn as nn
import torchvision.models as models

import resnet

'''
Basic block is the residual block described in the Resnet paper
https://arxiv.org/pdf/1512.03385.pdf
'''


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dilation=1):
        super(BasicBlock, self).__init__()

        # Layer 1
        """
            1) Resnet always uses a kernel size of 3x3
            2) Kernel size = 3 maps to padding = 1 to preserve input resolution
            3) Image resolution down sampling is supported in layer 1 using stride
            4) Bias is not used in Resnet
        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=dilation, stride=stride, dilation=dilation, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Layer 2
        """
            1) Layer 2 does not support changing the image resolution i.e stride=1
            2) Kernel size = 3 maps to padding = 1 to preserve input resolution
            3) Bias is not supported in Layer 2 as well
        """
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=dilation, stride=1, bias=False, dilation=dilation)

        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            '''
             In this case, layer 1 resolution is down sampled and the out_channels is doubled
             E.g
             Input: HxW, 32 channels, stride=2 
             After Layer 1: H/2 x W/2, 64 channels
             After Layer 2: H/2 x W/2, 64 channels
             In these case, the input cannot be added to output of Layer2 directly.
             To handle this case, 1x1 convolution with out_channels = 64 and 
             stride = 2 is used to convert the input to 320x320, 64 channels 
             which can bet added to the output of layer 2 
            '''
            self.conv_lifting = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_lifting = nn.Sequential()  # NO OP

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.conv_lifting(x)
        out = self.relu(out)
        return out


class UpsamplingBlock(nn.Module):

    def __init__(
            self,
            input: int,
            output: int,
            scale_factor: nn.Module = None,
            upsample_mode: str = "bilinear",
            norm_layer=None
    ) -> None:
        super(UpsamplingBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = resnet.conv3x3(input, output, stride=1)
        self.bn1 = norm_layer(output)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.upsample(out)
        return out


"""
    For PVNet, the following layers are re-used
    1) Conv-BN-Relu
    2) Max Pooling
    3) Layer1, Layer2, Layer 3
    
    The working of each of the layer is described below:
    1) Conv-BN-Relu
        a) Converts input image with 3 channels(RGB) to 64 channels
        b) kernel_size = 7 corresponds to padding = 3 to preserve the input dimension
        c) Stride=2 reduces image dimension by 2.
        Dim: H/2 x W/2, channels = 64 
        
    2) Max Pooling
        Max pooling reduces the image dimension again by half.
        Dim: H/4 x W/4, channels = 64
        
    3) Layer 1
        This residual layer maintains the image dimensions.
        Dim: H/4 x W/4, channels = 64
        
    4) Layer 2
        Residual blocks layer with strided convolution. The out_channels is doubled and the 
        image dimension is halved with stride=2. 
        Dim: H/8 x W/8, channels = 128
        
    5) Layer 3
        Residual blocks layer with strided convolution. The out_channels is doubled
        and the image dimension is halved with stride=2. 
        Dim: H/16 x W/16, channels = 256
"""


class Resnet18Modified(resnet.ResNet):
    def __init__(self, num_classes: int = 1000):
        super(Resnet18Modified, self).__init__(resnet.BasicBlock,
                                               [2, 2, 2, 2],  # Resnet18
                                               num_classes=num_classes)

        self.skip0 = None
        self.skip1 = None
        self.skip2 = None
        self.skip3 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.skip0 = x  # H/2 x W/2, channels = 64

        x = self.maxpool(x)  # H/4 x W/4, channels = 64

        x = self.layer1(x)
        self.skip1 = x  # H/4 x W/4, channels = 64

        x = self.layer2(x)
        self.skip2 = x   # H/8 x W/8, channels = 128

        x = self.layer3(x)
        self.skip3 = x  # H/16 x W/16, channels = 256

        return x


"""
PVNet is basically Resnet 18 architecture with modifications to the fully connected 
layer and custom bi-linear up sampling layers
"""


class PvNet(nn.Module):
    """
    Given C = num_classes, K = num_keypoints per class

    Take input RGB image, if 'output_class' then output semantic seg ((C+1)xHxW), if 'output_vector' then
    output pixel-wise keypoint vector ((C*K+1)xHxW). 18 because 2 channels for vector (u,v) pointing to each
    keypoint, for 9 keypoints

    The first half is a modified version of ResNet-18, but modifies layer4 and the final pooling/fc step
    so that the image dimensionality is not reduced below [H/8,W/8]

    We still try to initialize the first few layers with a pre-trained set of weights from canonical ResNet-18

    The below architecture is implemented based on
    1) Figure 2b in the following paper
    https://arxiv.org/pdf/1812.11788.pdf

    2) Table 1 in the following paper
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self,
                 num_classes: int = 12,
                 num_keypoints: int = 9,
                 norm_layer=None,
                 output_class=True,
                 output_vector=False,
                 pretrained=True):
        super(PvNet, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.output_class = output_class
        self.output_vector = output_vector

        # Initial layers of PVNet borrowed from resnet18
        self.resnet18 = Resnet18Modified(num_classes=self.num_classes)

        """
            Layer 4:
            Unlike ResNet 18, for layer 4 we don't striding by 2 so we stop down sampling image
            We dilate by 2x so the kernel still "covers" the same range of pixels it would stride = 2
            Dim: H/16 x W/16, channels = 512
        """
        self.layer4 = BasicBlock(in_channels=256, out_channels=512, stride=1, dilation=2)

        # TODO: Check with Chen if this is the right convolution
        self.reverseConv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        self.preUpsampleConv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        # Add upsampling layers
        self.upsample1 = UpsamplingBlock(256, 128, 2, upsample_mode="bilinear")
        self.upsample2 = UpsamplingBlock(128, 64, 2, upsample_mode="bilinear")
        self.upsample3 = UpsamplingBlock(64, 64, 2, upsample_mode="bilinear")

        # Final convolution before output layers
        self.upsample4 = UpsamplingBlock(64, 64, 2, upsample_mode="bilinear")

        """
        Output convolutions
        1x1 convolutions over full depth, reshaping to appropriate output
        For class, output num_classes+1 channels with probability for each class (including one null class)
        """
        if self.output_class:
            self.class_out = nn.Conv2d(64, num_classes + 1, kernel_size=1, stride=1, bias=False)

        """
        For vector, output num_classes * num_keypoints * 2, since each class can have num_keypoints,
        and each keypoint has [u,v]
        """
        if self.output_vector:
            self.vector_out = nn.Conv2d(64, num_keypoints * num_classes * 2, kernel_size=1, stride=1, bias=False)

        # TODO: Initialize all modules above with ResNet18 pre-trained weights, where they exist
        if pretrained:
            resnet18_pretrained = models.resnet18(pretrained=True)

            self.resnet18.conv1.load_state_dict(resnet18_pretrained.conv1.state_dict())
            self.resnet18.bn1.load_state_dict(resnet18_pretrained.bn1.state_dict())
            self.resnet18.relu.load_state_dict(resnet18_pretrained.relu.state_dict())
            self.resnet18.maxpool.load_state_dict(resnet18_pretrained.maxpool.state_dict())

            self.resnet18.layer1.load_state_dict(resnet18_pretrained.layer1.state_dict())
            self.resnet18.layer2.load_state_dict(resnet18_pretrained.layer2.state_dict())
            self.resnet18.layer3.load_state_dict(resnet18_pretrained.layer3.state_dict())

    def forward(self, x):

        x = self.resnet18.forward(x)

        x = self.layer4(x)  # H/16 x W/16, channels = 512

        x = self.reverseConv(x)  # H/16 x W/16, channels = 256
        x = self.preUpsampleConv(x)  # H/16 x W/16, channels = 256

        # First skip layer add back
        x = self.upsample1(x + self.resnet18.skip3)  # H/8 x W/8, channels = 128
        x = self.upsample2(x + self.resnet18.skip2)  # H/4 x W/4, channels = 64
        x = self.upsample3(x + self.resnet18.skip1)  # H/2 x W/2, channels = 64
        x = self.upsample4(x + self.resnet18.skip0)  # H x W, channels = 64

        outputs = {}

        if self.output_class:
            outputs['class'] = self.class_out(x)

        if self.output_vector:
            outputs['vector'] = self.vector_out(x)

        return outputs


if __name__ == '__main__':
    pvnet = PvNet(
        num_classes=12,
        num_keypoints=9,
        norm_layer=None,
        output_class=True,
        output_vector=False,
        pretrained=True
    )
