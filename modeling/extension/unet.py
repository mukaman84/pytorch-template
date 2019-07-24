import math
import torch.nn as nn
import torch
import numpy as np


class upconv(nn.Module):
    def __init__(self, features_in, features_out, BatchNorm, stride=1):
        super(upconv, self).__init__()
        self.conv = nn.ConvTranspose2d(features_in, features_out, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.bn = BatchNorm(features_out)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class skipmerger(nn.Module):
    def __init__(self, features_out, kernel_size, stride=1):
        super(skipmerger, self).__init__()
        self.conv = nn.Conv2d(2*features_out, features_out, kernel_size=kernel_size, stride=stride, padding=1)


    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, num_filters, kernel_size, padding, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,batchNormObject=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.conv2 = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation )
        self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        out = og_x + x
        out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out


class preconv(nn.Module):
    def __init__(self, features_in, features_out, BatchNorm, kernel_size=3, stride=1):
        super(preconv, self).__init__()
        self.conv = nn.Conv2d(features_in, features_out, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = BatchNorm(features_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_channels,num_hidden_features,kernel_size,padding,n_resblocks,BatchNorm,dropout_min=0,dropout_max=0.2,
                 upconvObject=upconv, skipObject=skipmerger, convObject=ResidualBlock,preconvObject=preconv):
        # self.inplanes = 64
        super(Decoder, self).__init__()

        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.ModuleList()
        self.skipmerger = nn.ModuleList()
        self.residualBlocks = nn.ModuleList()
        self.preconv = nn.ModuleList()

        dropout = iter([(1 - t) * dropout_min + t * dropout_max for t in np.linspace(0, 1, (len(num_hidden_features)))][::-1])



        for features_in, features_out, back_feature_in in [num_hidden_features[i:i + 2]+[j] for i, j in zip(range(0, len(num_hidden_features), 1),[512,256,64])]:
            self.upsample.append(upconvObject(features_in, features_out, BatchNorm, stride=2))
            self.skipmerger.append(skipObject(features_out, kernel_size, stride=1))
            self.preconv.append(preconvObject(back_feature_in, features_out, BatchNorm, kernel_size, stride=1))

            block = []
            p = next(iter(dropout))

            for _ in range(n_resblocks):
                block += [convObject(features_out, kernel_size, padding, dropout=p, batchNormObject=BatchNorm)]
            self.residualBlocks.append(nn.Sequential(*block))

        self.output_upconv =upconvObject(features_out, features_out, BatchNorm, stride=2)

        block = [nn.Conv2d(num_hidden_features[-1], 32, kernel_size=kernel_size, stride=1,padding=padding)]
        self.output_convolution = nn.Sequential(*block)

    # self._init_weight()
    #
    # def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             BatchNorm(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
    #
    #     return nn.Sequential(*layers)
    #
    # def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             BatchNorm(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
    #                         downsample=downsample, BatchNorm=BatchNorm))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, len(blocks)):
    #         layers.append(block(self.inplanes, planes, stride=1,
    #                             dilation=blocks[i]*dilation, BatchNorm=BatchNorm))
    #
    #     return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x,x2,x3,x4):

        for up,merge,pre,conv,skip in zip(self.upsample,self.skipmerger,self.preconv,self.residualBlocks,[x2,x3,x4]):
            x=up(x)
            skip=pre(skip)
            cat = torch.cat([x, skip], dim=1)
            x = merge(cat)
            x = conv(x)

        x = self.output_upconv(x)

        return self.output_convolution(x)





def UnetDecoder(out_channels,kernel_size,padding,n_resblocks):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    num_hidden_features = [32,64,128,256]


    model = Decoder(out_channels,num_hidden_features,kernel_size,padding,n_resblocks,BatchNorm=nn.BatchNorm2d,dropout_min=0,dropout_max=0.2,
                    upconvObject=upconv, skipObject=skipmerger, convObject=ResidualBlock,preconvObject=preconv)
    return model