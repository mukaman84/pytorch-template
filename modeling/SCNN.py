import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.backbone import build_backbone
from modeling.extension import build_extension
from modeling.build_rnn import build_rnn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SCNN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16,nclass = 19,cuda= True,extension=None):
        super(SCNN, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d
        if cuda == True:
            device = "cuda"
        else:
            device = "cpu"
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rnn = build_rnn(32,device)
        self.conv0 = nn.Sequential(nn.Conv2d(2048, 32, 3, padding=1, bias=False),
                                   BatchNorm(32),
                                    nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(32, nclass, 3, padding=1, bias=False),
                                   BatchNorm(nclass),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 2, 3, padding=1, bias=False),
                                   BatchNorm(2),
                                   nn.Softmax())
        self.dropout = nn.Dropout2d()
        self.extension = build_extension(ext=extension,out_channels=nclass, kernel_size=3, padding=1, n_resblocks=3)




    def forward(self, input):
        x1,x2,x3,x4 = self.backbone(input)
        x = x1.view(x1.size()).permute(0, 2, 3, 1)
        x = self.rnn(x)
        x = self.conv0(x)
        x = self.dropout(x)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = self.extension(x,x2,x3,x4)
        y = self.conv2(x)
        x = self.conv1(x)

        # a= y.data.cpu().numpy()

        return x, y


if __name__ == "__main__":

    inputs = torch.rand(2, 3, 512, 512)# N x C x H x W
    # N, C, H, W = x.size()
    # inputs = inputs.view(inputs.size()).permute(0, 2, 3, 1)
    # print("x.size():", inputs.size())


    # inputs = torch.rand(8, 36, 100, 128)# N x H x W x C
    channel = inputs.size()[1]
    model = SCNN("mobilenet")
    output = model(inputs)
    print(output.size())
