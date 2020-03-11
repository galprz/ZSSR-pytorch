import torch
from math import sqrt
import torch.nn as nn

class ZSSRModel(nn.Module):
    def __init__(self, input_channels=3, channels=64, ks=3, layers_num=8, device='cuda'):
        super(ZSSRModel, self).__init__()
        layers = [
            nn.Conv2d(input_channels, channels, kernel_size=ks, padding=ks // 2, bias=True),
            nn.ReLU()
        ]
        for _ in range(layers_num - 2):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=ks, padding=ks//2, bias=True),
                nn.ReLU()
            ]
        layers += [nn.Conv2d(channels, input_channels, kernel_size=ks, padding=ks//2, bias=True)]
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, X):
        return self.model(X) + X

class ZSSRModelWithBackbone(nn.Module):
    def __init__(self, backbone_model, freeze_backbone=True):
        super(ZSSRModelWithBackbone, self).__init__()
        self.zssr = ZSSRModel()
        self.backbone_model = backbone_model
        for p in backbone_model.parameters():
            p.requires_grad = not freeze_backbone
        self.backbone_model.train(not freeze_backbone)

    def forward(self, X):
        out = self.backbone_model(X)
        out = self.zssr(out)
        return out  + X

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out