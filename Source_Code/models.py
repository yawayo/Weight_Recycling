import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_recycle import Weight_recycle_Conv2d
import time


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin//4, kernel_size=kernel_size, padding=padding, groups=nin//4, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout//4, kernel_size=1, bias=bias)

    def forward(self, x):
        out_0 = self.depthwise(x)
        out_90 = self.depthwise(torch.rot90(x, 1, [2, 3]))
        out_180 = self.depthwise(torch.rot90(x, 2, [2, 3]))
        out_270 = self.depthwise(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out_0 = self.pointwise(out)
        out_90 = self.pointwise(torch.rot90(out, 1, [2, 3]))
        out_180 = self.pointwise(torch.rot90(out, 2, [2, 3]))
        out_270 = self.pointwise(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        return out

class Xception(nn.Module):
    def __init__(self, input_channel, num_classes=100):
        super(Xception, self).__init__()


        # Entry Flow
        self.entryflow_1_1_conv = nn.Conv2d(input_channel, 32//4, kernel_size=3, stride=2, padding=1, bias=False)
        self.entry_flow_1_1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.entryflow_1_2_conv = nn.Conv2d(32, 64//4, kernel_size=3, stride=1, padding=1)
        self.entry_flow_1_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )


        self.entryflow_2_1_conv = depthwise_separable_conv(64, 128, 3, 1)
        self.entry_flow_2_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.entryflow_2_2_conv = depthwise_separable_conv(128, 128, 3, 1)
        self.entry_flow_2_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.entry_flow_2_residual = nn.Conv2d(64, 128//4, kernel_size=1, stride=2, padding=0)


        self.entry_flow_3_1_relu = nn.ReLU(True)
        self.entry_flow_3_1_conv = depthwise_separable_conv(128, 256, 3, 1)
        self.entry_flow_3_1_bn = nn.BatchNorm2d(256)
        self.entry_flow_3_2_relu = nn.ReLU(True)
        self.entry_flow_3_2_conv = depthwise_separable_conv(256, 256, 3, 1)
        self.entry_flow_3_2_bn = nn.BatchNorm2d(256)
        self.entry_flow_3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.entry_flow_3_residual = nn.Conv2d(128, 256//4, kernel_size=1, stride=2, padding=0)


        self.entry_flow_4_1_relu = nn.ReLU(True)
        self.entry_flow_4_1_conv = depthwise_separable_conv(256, 728, 3, 1)
        self.entry_flow_4_1_bn = nn.BatchNorm2d(728)
        self.entry_flow_4_2_relu = nn.ReLU(True)
        self.entry_flow_4_2_conv = depthwise_separable_conv(728, 728, 3, 1)
        self.entry_flow_4_2_bn = nn.BatchNorm2d(728)
        self.entry_flow_4_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.entry_flow_4_residual = nn.Conv2d(256, 728//4, kernel_size=1, stride=2, padding=0)


        # Middle Flow
        self.entry_flow_mid_1_relu = nn.ReLU(True)
        self.entry_flow_mid_1_conv = depthwise_separable_conv(728, 728, 3, 1)
        self.entry_flow_mid_1_bn = nn.BatchNorm2d(728)
        self.entry_flow_mid_2_relu = nn.ReLU(True)
        self.entry_flow_mid_2_conv = depthwise_separable_conv(728, 728, 3, 1)
        self.entry_flow_mid_2_bn = nn.BatchNorm2d(728)
        self.entry_flow_mid_3_relu = nn.ReLU(True)
        self.entry_flow_mid_3_conv = depthwise_separable_conv(728, 728, 3, 1)
        self.entry_flow_mid_3_bn = nn.BatchNorm2d(728)


        # Exit Flow
        self.entry_flow_ex_1_relu = nn.ReLU(True)
        self.entry_flow_ex_1_conv = depthwise_separable_conv(728, 728, 3, 1)
        self.entry_flow_ex_1_bn = nn.BatchNorm2d(728)
        self.entry_flow_ex_2_relu = nn.ReLU(True)
        self.entry_flow_ex_2_conv = depthwise_separable_conv(728, 1024, 3, 1)
        self.entry_flow_ex_2_bn = nn.BatchNorm2d(1024)
        self.entry_flow_ex_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.exit_flow_1_residual = nn.Conv2d(728, 1024//4, kernel_size=1, stride=2, padding=0)


        self.entryflow_ex_1_conv = depthwise_separable_conv(1024, 1536, 3, 1)
        self.entry_flow_ex_1 = nn.Sequential(
            nn.BatchNorm2d(1536),
            nn.ReLU(True))
        self.entryflow_ex_2_conv = depthwise_separable_conv(1536, 2048, 3, 1)
        self.entry_flow_ex_2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        entry_out1_0 = self.entryflow_1_1_conv(x)
        entry_out1_90 = self.entryflow_1_1_conv(torch.rot90(x, 1, [2, 3]))
        entry_out1_180 = self.entryflow_1_1_conv(torch.rot90(x, 2, [2, 3]))
        entry_out1_270 = self.entryflow_1_1_conv(torch.rot90(x, 3, [2, 3]))
        entry_out1 = torch.cat((entry_out1_0, entry_out1_90, entry_out1_180, entry_out1_270), dim=1)
        entry_out1 = self.entry_flow_1_1(entry_out1)
        entry_out1_0 = self.entryflow_1_2_conv(entry_out1)
        entry_out1_90 = self.entryflow_1_2_conv(torch.rot90(entry_out1, 1, [2, 3]))
        entry_out1_180 = self.entryflow_1_2_conv(torch.rot90(entry_out1, 2, [2, 3]))
        entry_out1_270 = self.entryflow_1_2_conv(torch.rot90(entry_out1, 3, [2, 3]))
        entry_out1 = torch.cat((entry_out1_0, entry_out1_90, entry_out1_180, entry_out1_270), dim=1)
        entry_out1 = self.entry_flow_1_2(entry_out1)

        entry_out2 = self.entryflow_2_1_conv(entry_out1)
        entry_out2 = self.entry_flow_2_1(entry_out2)
        entry_out2 = self.entryflow_2_2_conv(entry_out2)
        entry_out2 = self.entry_flow_2_2(entry_out2)

        entry_residual_0 = self.entry_flow_2_residual(entry_out1)
        entry_residual_90 = self.entry_flow_2_residual(torch.rot90(entry_out1, 1, [2, 3]))
        entry_residual_180 = self.entry_flow_2_residual(torch.rot90(entry_out1, 2, [2, 3]))
        entry_residual_270 = self.entry_flow_2_residual(torch.rot90(entry_out1, 3, [2, 3]))
        entry_residual = torch.cat((entry_residual_0, entry_residual_90, entry_residual_180, entry_residual_270), dim=1)

        entry_out2 = entry_out2 + entry_residual

        entry_out3 = self.entry_flow_3_1_relu(entry_out2)
        entry_out3 = self.entry_flow_3_1_conv(entry_out3)
        entry_out3 = self.entry_flow_3_1_bn(entry_out3)
        entry_out3 = self.entry_flow_3_2_relu(entry_out3)
        entry_out3 = self.entry_flow_3_2_conv(entry_out3)
        entry_out3 = self.entry_flow_3_2_bn(entry_out3)
        entry_out3 = self.entry_flow_3_pool(entry_out3)

        entry_residual_0 = self.entry_flow_3_residual(entry_out2)
        entry_residual_90 = self.entry_flow_3_residual(torch.rot90(entry_out2, 1, [2, 3]))
        entry_residual_180 = self.entry_flow_3_residual(torch.rot90(entry_out2, 2, [2, 3]))
        entry_residual_270 = self.entry_flow_3_residual(torch.rot90(entry_out2, 3, [2, 3]))
        entry_residual = torch.cat((entry_residual_0, entry_residual_90, entry_residual_180, entry_residual_270), dim=1)

        entry_out3 = entry_out3 + entry_residual

        entry_out4 = self.entry_flow_4_1_relu(entry_out3)
        entry_out4 = self.entry_flow_4_1_conv(entry_out4)
        entry_out4 = self.entry_flow_4_1_bn(entry_out4)
        entry_out4 = self.entry_flow_4_2_relu(entry_out4)
        entry_out4 = self.entry_flow_4_2_conv(entry_out4)
        entry_out4 = self.entry_flow_4_2_bn(entry_out4)
        entry_out4 = self.entry_flow_4_pool(entry_out4)

        entry_residual_0 = self.entry_flow_4_residual(entry_out3)
        entry_residual_90 = self.entry_flow_4_residual(torch.rot90(entry_out3, 1, [2, 3]))
        entry_residual_180 = self.entry_flow_4_residual(torch.rot90(entry_out3, 2, [2, 3]))
        entry_residual_270 = self.entry_flow_4_residual(torch.rot90(entry_out3, 3, [2, 3]))
        entry_residual = torch.cat((entry_residual_0, entry_residual_90, entry_residual_180, entry_residual_270), dim=1)

        entry_out = entry_out4 + entry_residual

        middle_out = self.entry_flow_mid_1_relu(entry_out)
        middle_out = self.entry_flow_mid_1_conv(middle_out)
        middle_out = self.entry_flow_mid_1_bn(middle_out)
        middle_out = self.entry_flow_mid_2_relu(middle_out)
        middle_out = self.entry_flow_mid_2_conv(middle_out)
        middle_out = self.entry_flow_mid_2_bn(middle_out)
        middle_out = self.entry_flow_mid_3_relu(middle_out)
        middle_out = self.entry_flow_mid_3_conv(middle_out)
        middle_out = self.entry_flow_mid_3_bn(middle_out)
        middle_out = middle_out + entry_out

        exit_out1 = self.entry_flow_ex_1_relu(middle_out)
        exit_out1 = self.entry_flow_ex_1_conv(exit_out1)
        exit_out1 = self.entry_flow_ex_1_bn(exit_out1)
        exit_out1 = self.entry_flow_ex_2_relu(exit_out1)
        exit_out1 = self.entry_flow_ex_2_conv(exit_out1)
        exit_out1 = self.entry_flow_ex_2_bn(exit_out1)
        exit_out1 = self.entry_flow_ex_pool(exit_out1)

        exit_residual_0 = self.exit_flow_1_residual(middle_out)
        exit_residual_90 = self.exit_flow_1_residual(torch.rot90(middle_out, 1, [2, 3]))
        exit_residual_180 = self.exit_flow_1_residual(torch.rot90(middle_out, 2, [2, 3]))
        exit_residual_270 = self.exit_flow_1_residual(torch.rot90(middle_out, 3, [2, 3]))
        exit_residual = torch.cat((exit_residual_0, exit_residual_90, exit_residual_180, exit_residual_270), dim=1)

        exit_out1 = exit_out1 + exit_residual

        exit_out2 = self.entryflow_ex_1_conv(exit_out1)
        exit_out2 = self.entry_flow_ex_1(exit_out2)
        exit_out2 = self.entryflow_ex_2_conv(exit_out2)
        exit_out2 = self.entry_flow_ex_2(exit_out2)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)

        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.inplanes = in_planes
        self.planes = planes
        self.shortcut_conv = nn.Conv2d(in_planes, (self.expansion*planes)//4, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)

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
        if self.stride != 1 or self.inplanes != self.expansion*self.planes:
            out_0 = self.shortcut_conv(x)
            out_90 = self.shortcut_conv(torch.rot90(x, 1, [2, 3]))
            out_180 = self.shortcut_conv(torch.rot90(x, 2, [2, 3]))
            out_270 = self.shortcut_conv(torch.rot90(x, 3, [2, 3]))
            out_short = torch.cat((out_0, out_90, out_180, out_270), dim=1)
            out += self.shortcut_bn(out_short)
        else:
            out += x
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
        self.stride = stride
        self.inplanes = in_planes
        self.planes = planes
        self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes//4, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
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
        out_0 = self.conv3(out)
        out_90 = self.conv3(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.bn3(out)
        if self.stride != 1 or self.inplanes != self.expansion*self.planes:
            out_0 = self.shortcut_conv(x)
            out_90 = self.shortcut_conv(torch.rot90(x, 1, [2, 3]))
            out_180 = self.shortcut_conv(torch.rot90(x, 2, [2, 3]))
            out_270 = self.shortcut_conv(torch.rot90(x, 3, [2, 3]))
            out_short = torch.cat((out_0, out_90, out_180, out_270), dim=1)
            out += self.shortcut_bn(out_short)
        else:
            out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64//4, kernel_size=3, stride=1, padding=1, bias=False)
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
        out_0 = self.conv1(x)
        out_90 = self.conv1(torch.rot90(x, 1, [2, 3]))
        out_180 = self.conv1(torch.rot90(x, 2, [2, 3]))
        out_270 = self.conv1(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1)
        self.BN1_1 = nn.BatchNorm2d(64)
        self.BN1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1)
        self.BN2_1 = nn.BatchNorm2d(128)
        self.BN2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(256, 64, kernel_size=(3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(256, 64, kernel_size=(3, 3), padding=1)
        self.conv3_4 = nn.Conv2d(256, 64, kernel_size=(3, 3), padding=1)
        self.BN3_1 = nn.BatchNorm2d(256)
        self.BN3_2 = nn.BatchNorm2d(256)
        self.BN3_3 = nn.BatchNorm2d(256)
        self.BN3_4 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.conv4_3 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.conv4_4 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.BN4_1 = nn.BatchNorm2d(512)
        self.BN4_2 = nn.BatchNorm2d(512)
        self.BN4_3 = nn.BatchNorm2d(512)
        self.BN4_4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.conv5_3 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.conv5_4 = nn.Conv2d(512, 128, kernel_size=(3, 3), padding=1)
        self.BN5_1 = nn.BatchNorm2d(512)
        self.BN5_2 = nn.BatchNorm2d(512)
        self.BN5_3 = nn.BatchNorm2d(512)
        self.BN5_4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out_0 = self.conv1_1(x)
        out_90 = self.conv1_1(torch.rot90(x, 1, [2, 3]))
        out_180 = self.conv1_1(torch.rot90(x, 2, [2, 3]))
        out_270 = self.conv1_1(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN1_1(out)
        out = self.relu(out)

        out_0 = self.conv1_2(out)
        out_90 = self.conv1_2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv1_2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv1_2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN1_2(out)
        out = self.relu(out)

        out = F.max_pool2d(out, 2)

        out_0 = self.conv2_1(out)
        out_90 = self.conv2_1(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv2_1(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv2_1(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN2_1(out)
        out = self.relu(out)

        out_0 = self.conv2_2(out)
        out_90 = self.conv2_2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv2_2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv2_2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN2_2(out)
        out = self.relu(out)

        out = F.max_pool2d(out, 2)

        out_0 = self.conv3_1(out)
        out_90 = self.conv3_1(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3_1(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3_1(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN3_1(out)
        out = self.relu(out)

        out_0 = self.conv3_2(out)
        out_90 = self.conv3_2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3_2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3_2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN3_2(out)
        out = self.relu(out)

        out_0 = self.conv3_3(out)
        out_90 = self.conv3_3(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3_3(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3_3(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN3_3(out)
        out = self.relu(out)

        out_0 = self.conv3_4(out)
        out_90 = self.conv3_4(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv3_4(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv3_4(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN3_4(out)
        out = self.relu(out)

        out = F.max_pool2d(out, 2)

        out_0 = self.conv4_1(out)
        out_90 = self.conv4_1(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv4_1(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv4_1(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN4_1(out)
        out = self.relu(out)

        out_0 = self.conv4_2(out)
        out_90 = self.conv4_2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv4_2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv4_2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN4_2(out)
        out = self.relu(out)

        out_0 = self.conv4_3(out)
        out_90 = self.conv4_3(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv4_3(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv4_3(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN4_3(out)
        out = self.relu(out)

        out_0 = self.conv4_4(out)
        out_90 = self.conv4_4(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv4_4(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv4_4(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN4_4(out)
        out = self.relu(out)

        out = F.max_pool2d(out, 2)

        out_0 = self.conv5_1(out)
        out_90 = self.conv5_1(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv5_1(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv5_1(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN5_1(out)
        out = self.relu(out)

        out_0 = self.conv5_2(out)
        out_90 = self.conv5_2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv5_2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv5_2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN5_2(out)
        out = self.relu(out)

        out_0 = self.conv5_3(out)
        out_90 = self.conv5_3(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv5_3(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv5_3(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN5_3(out)
        out = self.relu(out)

        out_0 = self.conv5_4(out)
        out_90 = self.conv5_4(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv5_4(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv5_4(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = self.BN5_4(out)
        out = self.relu(out)

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, (5, 5))
        self.conv2 = nn.Conv2d(16, 16, (5, 5))
        self.fc1   = nn.Linear(64*5*5, 120)
        self.fc2   = nn.Linear(120, 600)
        self.fc3   = nn.Linear(600, 100)

    def forward(self, x):
        out_0 = self.conv1(x)
        out_90 = self.conv1(torch.rot90(x, 1, [2, 3]))
        out_180 = self.conv1(torch.rot90(x, 2, [2, 3]))
        out_270 = self.conv1(torch.rot90(x, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out_0 = self.conv2(out)
        out_90 = self.conv2(torch.rot90(out, 1, [2, 3]))
        out_180 = self.conv2(torch.rot90(out, 2, [2, 3]))
        out_270 = self.conv2(torch.rot90(out, 3, [2, 3]))
        out = torch.cat((out_0, out_90, out_180, out_270), dim=1)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out