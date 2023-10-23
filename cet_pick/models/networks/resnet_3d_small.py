import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights_3d(up):
    w = up.weight.data
    # print('w', w.size())
    f = math.ceil(w.size(2) / 2)
    # print('f', f)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            for k in range(w.size(4)):
                w[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))*(1 - math.fabs(k / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :, :] = w[0, 0, :, :, :] 

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TomoRes3DNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoRes3DNet, self).__init__()
        print('using 3d mode')
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        # self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2)
        self.deconv_layers = self._make_deconv_layer(
            2,
            [64, 32],
            [4, 4],
        )

        self.feature_head = nn.Sequential(nn.Conv3d(32, 16, kernel_size = (3,3,3), padding=(1,1,1), bias=True), 
                    nn.ReLU(inplace=True)
            )
        fill_fc_weights(self.feature_head)
        # self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
        # self.feature_head = nn.Sequential()
        for head in self.heads:
            # print('head', head)
            # head_out = head + '_out'
            classes = self.heads[head]
            # if head_conv > 0:
            #     fc = nn.Sequential(
            #         nn.Conv3d(32, head_conv, kernel_size = 3, padding=1, bias=True), 
            #         nn.ReLU(inplace=True)
            #         )
            #     fl = nn.Conv3d(head_conv, classes, kernel_size = 1, stride=1, padding=0, bias=True)
            #     if 'hm' in head:
            #         fl.bias.data.fill_(-2.19)
            #     # else:
            #     fill_fc_weights(fc)
            # else:
            # this can be viewed as projection head in contrastive learning
            fc = nn.Conv3d(16, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)



    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
                # nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation = dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation = dilation))

        return nn.Sequential(*layers)


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes, 
            #         kernel_size=(3,3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv3d(self.inplanes, planes,
                    kernel_size=3, stride=1, 
                    padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
            up = nn.ConvTranspose3d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights_3d(up)

            layers.append(fc)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.deconv_layers(x)

        x = self.feature_head(x)


        ret = {}
        for head in self.heads:

            out = self.__getattr__(head)(x)
            if 'proj' in head:

                out = torch.nn.functional.normalize(out, dim=1)

            ret[head] = out

            print('out', ret[head].shape)

        return [ret]

    def _load_pretrained(self, state_dict, inchans=3):
        if inchans == 1:
            conv1_weights = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"
        self.load_state_dict(state_dict, strict=False)
    def init_weights(self, num_layers):
        if 1:
            # url = model_urls['resnet{}'.format(num_layers)]

            # pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            checkpoint = torch.load('/hpc/home/qh36/research/qh36/3D_picking/cet_pick/cet_pick/models/resnet_18.pth')
            pretrained_state_dict = checkpoint['state_dict']
            # print('pretrained_state_dict', pretrained_state_dict)
            # state_dict = pretrained_state_dict.copy()
            for k in pretrained_state_dict.copy():
                if k.startswith('module') and not k.startswith('module_list'):
                    pretrained_state_dict[k[7:]] = pretrained_state_dict[k]
                else:
                    pretrained_state_dict[k] = pretrained_state_dict[k]
            # self.load_state_dict(pretrained_state_dict, strict=False)
            self._load_pretrained(pretrained_state_dict, inchans=1)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_tomo_net_3d(num_layers, heads, head_conv = 16):

    block_class, layers = resnet_spec[num_layers]

    model = TomoRes3DNet(block_class, layers, heads, head_conv=16)
    # model.init_weights(num_layers)
    return model
