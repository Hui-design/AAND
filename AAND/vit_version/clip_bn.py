import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import os, pdb, sys
sys.path.append(os.getcwd())
from typing import Type, Any, Callable, Union, List, Optional
from models.recons_net import RAR_single, MLP
from models.resnet_rar import BasicBlock, Bottleneck, AttnBottleneck, conv1x1, conv3x3
import pdb


class BN_layer(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 768
        self.dilation = 1
        # self.bn_layer = self._make_layer(block, 768, layers, stride=1)

        self.conv1 = conv1x1(768, 768, 1)
        self.bn1 = norm_layer(768)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(768, 768, 1)
        self.bn2 = norm_layer(768)
        self.conv3 = conv1x1(768, 768, 1)
        self.bn3 = norm_layer(768)
        # self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        # self.bn3 = norm_layer(256 * block.expansion)

        self.conv4 = conv1x1(768*3, 768, 1)
        self.bn4 = norm_layer(768)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
    #                 stride: int = 1, dilate: bool = False) -> nn.Sequential:
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #     #                     self.base_width, previous_dilation, norm_layer))
    #     # self.inplanes = planes * block.expansion
    #     self.inplances = planes
    #     for _ in range(blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #x = self.cbam(x)
        l1 = self.relu(self.bn1(self.conv1(x[0])))
        l2 = self.relu(self.bn2(self.conv2(x[1])))
        l3 = self.relu(self.bn3(self.conv3(x[2])))
        feature = torch.cat([l1,l2,l3],1)
        output = self.conv4(feature)
        # output = self.bn_layer(feature)
        #x = self.avgpool(feature_d)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if __name__ == '__main__':
    x = [torch.randn((1,768,16,16))]*3
    bn = BN_layer(AttnBottleneck, 3)
    pdb.set_trace()
    x_new = bn(x)