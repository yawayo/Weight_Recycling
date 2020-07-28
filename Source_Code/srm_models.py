import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from srm_block import SRMLayer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut_conv = nn.Conv2d(in_planes, (self.expansion*planes)//4, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)

        self.srm_block = SRMLayer(planes)

    def forward(self, x):
        out_0 = self.conv1(x)
        out_90 = self.conv1(torch.rot90(x, 1, [2, 3]))
        out_180 = self.conv1(torch.rot90(x, 2, [2, 3]))
        out_270 = self.conv1(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(self.bn1(out))
        out_0 = self.conv2(out)
        out_90 = self.conv2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.bn2(out)
        out = self.srm_block(out)
        out_0 = self.shortcut_conv(x)
        out_90 = self.shortcut_conv(torch.rot90(x, 1, [2, 3]))
        out_180 = self.shortcut_conv(torch.rot90(x, 2, [2, 3]))
        out_270 = self.shortcut_conv(torch.rot90(x, 3, [2, 3]))
        out_short = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out += self.shortcut_bn(out_short)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes//4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes//4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes//4, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)

        self.srm_block = SRMLayer(self.expansion*planes)

    def forward(self, x):
        out_0 = self.conv1(x)
        out_90 = self.conv1(torch.rot90(x, 1, [2, 3]))
        out_180 = self.conv1(torch.rot90(x, 2, [2, 3]))
        out_270 = self.conv1(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(self.bn1(out))
        out_0 = self.conv2(out)
        out_90 = self.conv2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(self.bn2(out))
        out_0 = self.conv3(out)
        out_90 = self.conv3(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.bn3(out)
        out = self.srm_block(out)
        out_0 = self.shortcut_conv(x)
        out_90 = self.shortcut_conv(torch.rot90(x, 1, [2, 3]))
        out_180 = self.shortcut_conv(torch.rot90(x, 2, [2, 3]))
        out_270 = self.shortcut_conv(torch.rot90(x, 3, [2, 3]))
        out_short = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out += self.shortcut_bn(out_short)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_srm():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34_srm():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50_srm():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101_srm():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152_srm():
    return ResNet(Bottleneck, [3,8,36,3])