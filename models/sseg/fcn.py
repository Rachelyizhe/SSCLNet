#-----------------------------------------------------------------------------------
#                            Sem+Bin
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_bin = PSPHead(in_channels, 1, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1 = self.head_sem(x1)
#         out2 = self.head_sem(x2)

#         out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         return out1, out2, out_bin.squeeze(1)


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)



#-----------------------------------------------------------------------------------
#                            Scd
#-----------------------------------------------------------------------------------
from models.block.conv import conv3x3
from models.sseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class FCN(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(FCN, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1]
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.head_scd = FCNHead(in_channels, nclass, lightweight)

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
        x2 = self.backbone.base_forward(x2)[-1]

        out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
        out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

        out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
        out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

        return out1_scd, out2_scd


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1, bias=True))

    def forward(self, x):
        return self.head(x)


#-----------------------------------------------------------------------------------
#                            RecolBinLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]

#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.representation = nn.Sequential(conv3x3(in_channels, in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels // 4, 256, 1, bias=True))


#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]
#         out_bin = torch.abs(x1 - x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x2), 1)))
        
#         out_bin_rep = self.representation(out_bin)

#         return out1_scd, out2_scd, out_bin_rep



# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


#-----------------------------------------------------------------------------------
#                            RecolBinLoss+ScdLoss+Semloss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]

#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.representation = nn.Sequential(conv3x3(in_channels, in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels // 4, 256, 1, bias=True))


#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x1)

#         out_bin = torch.abs(x1 - x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x2), 1)))
        
#         out_bin_rep = self.representation(out_bin)

#         return out1_scd, out2_scd, out_bin_rep, out1_sem, out2_sem


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# -----------------------------------------------------------------------------------
#                            RecolBinLoss(PSP)+ScdLoss
# -----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]

#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.representation = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels, lightweight),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels, 256, 1)
#         )

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape
#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]
#         out_bin = torch.abs(x1 - x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x2), 1)))
        
#         out_bin1 = self.PyramidPooling(out_bin)
#         out_bin_rep = self.representation(out_bin1)
#         return out1_scd, out2_scd, out_bin_rep



# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)












#-----------------------------------------------------------------------------------
#                            Sem+Scd
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_scd = FCNHead(in_channels, nclass, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x2)

#         out1_sem = F.interpolate(out1_sem, size=(h, w), mode='bilinear', align_corners=False)
#         out2_sem = F.interpolate(out2_sem, size=(h, w), mode='bilinear', align_corners=False)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)


#         return out1_scd, out2_scd, out1_sem, out2_sem


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)



#-----------------------------------------------------------------------------------
#                            Sem+Scd+Semrep
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.head_sem_rep = FCNHead(in_channels, 256, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x2)

#         out1_sem = F.interpolate(out1_sem, size=(h, w), mode='bilinear', align_corners=False)
#         out2_sem = F.interpolate(out2_sem, size=(h, w), mode='bilinear', align_corners=False)

#         out1_sem_rep = self.head_sem_rep(x1)
#         out2_sem_rep = self.head_sem_rep(x2)

#         out1_sem_rep = F.interpolate(out1_sem_rep, size=(128, 128), mode='bilinear', align_corners=False)
#         out2_sem_rep = F.interpolate(out2_sem_rep, size=(128, 128), mode='bilinear', align_corners=False)


#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)


#         return out1_scd, out2_scd, out1_sem, out2_sem, out1_sem_rep, out2_sem_rep


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)




#-----------------------------------------------------------------------------------
#                            Sem+Scd+Semrep+10184
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.head_sem_rep =  FCNHead(in_channels, 256, lightweight)

#         # self.head_sem_rep = nn.Sequential(conv3x3(in_channels, in_channels // 4, lightweight),
#         #                           nn.BatchNorm2d(in_channels // 4),
#         #                           nn.ReLU(True),
#         #                           nn.Conv2d(in_channels // 4, 256, 1, bias=True))

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x2)

#         out1_sem = F.interpolate(out1_sem, size=(h, w), mode='bilinear', align_corners=False)
#         out2_sem = F.interpolate(out2_sem, size=(h, w), mode='bilinear', align_corners=False)

#         out1_sem_rep = self.head_sem_rep(x1)
#         out2_sem_rep = self.head_sem_rep(x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)


#         return out1_scd, out2_scd, out1_sem, out2_sem, out1_sem_rep, out2_sem_rep


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)








# -----------------------------------------------------------------------------------
#                            Sem+Scd+binrep
# -----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_scd = FCNHead(in_channels, nclass, lightweight)
#         self.head_bin_rep = nn.Sequential(conv3x3(in_channels, in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels // 4, 256, 1, bias=True))


#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x2)

#         out1_sem = F.interpolate(out1_sem, size=(h, w), mode='bilinear', align_corners=False)
#         out2_sem = F.interpolate(out2_sem, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((out_bin, x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin_rep = self.head_bin_rep(out_bin)

#         return out1_scd, out2_scd, out1_sem, out2_sem, out_bin_rep


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)






# -----------------------------------------------------------------------------------
#                            Sem+Bin
# -----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_bin = PSPHead(in_channels, 1, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]


#         out1 = self.head_sem(x1)
#         out2 = self.head_sem(x2)

#         out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = torch.sigmoid(out_bin)
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)

#         return out1, out2, out_bin.squeeze(1)


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)

#-----------------------------------------------------------------------------------
#                            Sem+Bin+rep
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_bin = PSPHead(in_channels, 1, lightweight)
#         self.head_sem_rep = FCNHead(in_channels, 256, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1 = self.head_sem(x1)
#         out2 = self.head_sem(x2)

#         out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

#         out1_rep = self.head_sem_rep(x1)
#         out2_rep = self.head_sem_rep(x2)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = torch.sigmoid(out_bin)
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)

#         return out1, out2, out_bin.squeeze(1), out1_rep, out2_rep


# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)


# class PSPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(PSPHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.conv5 = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True),
#                                    nn.Dropout(0.1, False),
#                                    nn.Conv2d(inter_channels, out_channels, 1))

#     def forward(self, x):
#         return self.conv5(x)


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU(True))

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)





#-----------------------------------------------------------------------------------
#                            Sem+bin
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class FCN(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(FCN, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.head = FCNHead(in_channels, nclass-1, lightweight)
#         self.head_bin = FCNHead(in_channels, 1, lightweight)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1 = self.head(x1)
#         out2 = self.head(x2)

#         out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin)
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         return out1, out2, out_bin.squeeze(1)
   

# class FCNHead(nn.Module):
#     def __init__(self, in_channels, out_channels, lightweight):
#         super(FCNHead, self).__init__()
#         inter_channels = in_channels // 4

#         self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False),
#                                   nn.Conv2d(inter_channels, out_channels, 1, bias=True))

#     def forward(self, x):
#         return self.head(x)

