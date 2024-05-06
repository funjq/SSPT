import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=384):  # 384 1
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)


        return [P3_x, P4_x, P5_x]






class conv(nn.Module):
    def __init__(self, indim, outdim, kerner=3, stride=1):
        super(conv, self).__init__()
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=kerner, stride=stride, padding=kerner//2,dilation=1,bias=True)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(outdim)

    def forward(self, x):
        return (self.act(self.conv1(x)))
        # return self.conv1(x)
class CustomUpsampler(nn.Module):
    def __init__(self):
        super(CustomUpsampler, self).__init__()


        self.conv1 = conv(320, 160)
        self.conv2 = conv(160, 80)
        self.conv3 = conv(80, 40)
        self.conv4 = conv(40, 20)
        self.conv5 = nn.Conv2d(20, 1,kernel_size=1)

        self.adjust1 = conv(320, 80)
        self.adjust2 = conv(320, 40)

        self.adjust3 = nn.Sequential(
            conv(80, 40),
            conv(40, 1),
        )
        self.adjust4 = nn.Sequential(

            conv(40, 1),
        )

    def forward(self, x):
        # Top-left branch
        x_tl1 = self.conv1(x)
        x_tl2 = self.conv2(x_tl1)

        # Up-1
        x_init_up1 = F.interpolate(self.adjust1(x), scale_factor=4,mode='bilinear')
        x_up1 = F.interpolate(x_tl2, scale_factor=4,mode='bilinear')
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3(x_up1)

        # Up-2
        x_init_up2 = F.interpolate(self.adjust2(x), scale_factor=16,mode='bilinear')
        x_up2 = F.interpolate(x_tl3, scale_factor=4,mode='bilinear')
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4(x_up2)
        score_map_tl = self.conv5(x_tl4) + F.interpolate(self.adjust3(x_tl2), scale_factor=16,mode='bilinear') + F.interpolate(self.adjust4(x_tl3),scale_factor=4,mode='bilinear')
        return score_map_tl








