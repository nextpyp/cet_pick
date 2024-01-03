import torch
import torch.nn as nn
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, downsample):
        super(BasicBlock, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1), 
                                  nn.BatchNorm2d(planes), nn.ReLU(), 
                                  nn.Dropout(p=dropout_rate),
                                  nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
        self.downsample = downsample

    def forward(self, x):
        bn_x = self.bn(x)
        out = self.conv(bn_x)
        if self.downsample: out += self.downsample(bn_x)
        else: out += x
        return out

    
class WRN(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, in_planes=3, num_classes=10):
        super(WRN, self).__init__()
        self.in_planes = in_planes
        N, k = (depth - 4) // 6, widen_factor

        layers = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(self.in_planes, layers[0], kernel_size=3, stride=1, padding=1)
        self.in_planes = layers[0]
        self.conv2 = self._make_layer(layers[1], N, dropout_rate, stride=1)
        self.conv3 = self._make_layer(layers[2], N, dropout_rate, stride=2)
        self.conv4 = self._make_layer(layers[3], N, dropout_rate, stride=2)
        self.bn = nn.Sequential(nn.BatchNorm2d(layers[3]), nn.ReLU())
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(layers[3], num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, dropout_rate, stride):
        layers = []
        downsample = nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride)
        layers.append(BasicBlock(self.in_planes, planes, dropout_rate, stride, downsample))
        
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, dropout_rate, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


depth = 28
widen_factor = 10
dropout_rate = 0.3
num_classes = 10
model = WRN(depth,widen_factor,dropout_rate)
print('wider resnet', model)
pretrained_state_dict_path = '/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/models/pretrain/wideresnet10.pth'
pretrained_dict = torch.load(pretrained_state_dict_path,map_location='cuda:0')
print(pretrained_dict)
