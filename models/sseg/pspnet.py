#-----------------------------------------------------------------------------------
#                            Sem+Scd+Semrep
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem = PSPHead(in_channels, nclass-1, lightweight)
#         self.head_scd = PSPHead(in_channels, nclass, lightweight)
#         self.head_sem_rep = nn.Sequential(PyramidPooling(in_channels),
#                                    conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#                                    nn.BatchNorm2d(in_channels // 4),
#                                    nn.ReLU(True),
#                                    nn.Conv2d(in_channels // 4, 256, 1))

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






#-----------------------------------------------------------------------------------
#                           SemLoss+BinLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = PSPHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_seg = self.head_seg(x1)# [2,6,128,128]
#         out2_seg = self.head_seg(x2)

#         out1_seg = F.interpolate(out1_seg, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,6,512,512]
#         out2_seg = F.interpolate(out2_seg, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         return out1_seg, out2_seg, out_bin.squeeze(1)

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
#                           SemLoss(FCN)+BinLoss(PSP)+ScdLoss(PSP)
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = FCNHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 
#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_seg = self.head_seg(x1)# [2,6,128,128]
#         out2_seg = self.head_seg(x2)

#         out1_seg = F.interpolate(out1_seg, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,6,512,512]
#         out2_seg = F.interpolate(out2_seg, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out1_seg, out2_seg, out_bin.squeeze(1), out1_scd, out2_scd

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
#                           SemLoss(FCN)+BinLoss(PSP)
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = FCNHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_seg = self.head_seg(x1)# [2,6,128,128]
#         out2_seg = self.head_seg(x2)

#         out1_seg = F.interpolate(out1_seg, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,6,512,512]
#         out2_seg = F.interpolate(out2_seg, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         return out1_seg, out2_seg, out_bin.squeeze(1)

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
#                           SemLoss(PSP)+BinLoss(PSP)+Fusion
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = PSPHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 
#         self.conv1 = nn.Conv2d(64, nclass-1, 1, bias=False)
#         self.att = ChannelAttentionHL(nclass-1, nclass-1)
#         self.conv2 = nn.Conv2d(2*(nclass-1), nclass-1, 1, bias=False)
#         self.conv3 = nn.Conv2d(64, 1, 1, bias=False)
#         self.att_bin = ChannelAttentionHL(1, 1)
#         self.conv4 = nn.Conv2d(2, 1, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         # x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         # x2 = self.backbone.base_forward(x2)[-1]

#         x10, x1 = self.backbone.base_forward(x1) # x10=[2,64,256,256],x1=[2,600,128,128]
#         x20, x2 = self.backbone.base_forward(x2)


#         out1_seg = self.head_seg(x1)# [2,6,128,128]
#         out2_seg = self.head_seg(x2)

#         out1_seg_256_high = F.interpolate(out1_seg, size=(256, 256), mode='bilinear', align_corners=False) # out1=[2,6,256,256]
#         out2_seg_256_high = F.interpolate(out2_seg, size=(256, 256), mode='bilinear', align_corners=False)

#         out1_seg_256_low = self.att(out1_seg_256_high) * self.conv1(x10)
#         out2_seg_256_low = self.att(out2_seg_256_high) * self.conv1(x20)

#         out1_seg_256 = self.conv2(torch.cat((out1_seg_256_high, out1_seg_256_low), 1))
#         out2_seg_256 = self.conv2(torch.cat((out2_seg_256_high, out2_seg_256_low), 1))

#         out1_seg_512 = F.interpolate(out1_seg_256, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,6,512,512]
#         out2_seg_512 = F.interpolate(out2_seg_256, size=(h, w), mode='bilinear', align_corners=False)

#         out_bin = torch.abs(x1 - x2)
#         out_bin0 = torch.abs(x10 - x20)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin_256_high = F.interpolate(out_bin, size=(256, 256), mode='bilinear', align_corners=False)
#         out_bin_256_low = self.att_bin(out_bin_256_high) * self.conv3(out_bin0)
#         out_bin_256 = self.conv4(torch.cat((out_bin_256_high, out_bin_256_low), 1))
#         out_bin_512 = F.interpolate(out_bin_256, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin_512 = torch.sigmoid(out_bin_512)

#         return out1_seg_512, out2_seg_512, out_bin_512.squeeze(1)

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


# class ChannelAttentionHL(nn.Module):
#     def __init__(self, high_ch, low_ch):
#         super(ChannelAttentionHL, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc = nn.Conv2d(high_ch, low_ch, 1, bias=False)
#         # self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#16 is too large for remote sensing images?
#         # self.relu1 = nn.ReLU()
#         # self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
#         # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
#         avg_out=self.fc(self.avg_pool(x))#seem to work better SE-like attention
#         max_out=self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         #out = avg_out
#         return self.sigmoid(out)



#-----------------------------------------------------------------------------------
#                           ScdLoss+SemLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = FCNHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 
#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_seg = self.head_seg(x1)# [2,6,128,128]
#         out2_seg = self.head_seg(x2)

#         out1_seg = F.interpolate(out1_seg, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,6,512,512]
#         out2_seg = F.interpolate(out2_seg, size=(h, w), mode='bilinear', align_corners=False)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out1_seg, out2_seg, out1_scd, out2_scd

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
#                            RecolBinLoss+BinLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.classifier_bin = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels // 4, 2, 1)
#         )
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

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out_bin0 = torch.abs(x1 - x2)
#         out_bin1 = self.PyramidPooling(out_bin0)
#         out_bin_rep = self.representation(out_bin1)
#         out_bin = self.classifier_bin(out_bin1)
#         return out1_scd, out2_scd, out_bin, out_bin_rep


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
#                            RecolBinLoss+ScdLoss+Sem
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.head_sem = PSPHead(in_channels, nclass-1, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.classifier_bin = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels // 4, 2, 1)
#         )
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

#         out1_sem = self.head_sem(x1)
#         out2_sem = self.head_sem(x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out_bin0 = torch.abs(x1 - x2)
#         out_bin1 = self.PyramidPooling(out_bin0)
#         out_bin_rep = self.representation(out_bin1)
#         # out_bin = self.classifier_bin(out_bin1)
#         # return out1_scd, out2_scd, out_bin, out_bin_rep
#         return out1_scd, out2_scd, out_bin_rep, out1_sem, out2_sem

# #------------Spatial attention module in CBAM------------------------------------
# class SCSEBlock(nn.Module):
#     def __init__(self, channel):
#         super(SCSEBlock, self).__init__()

#         self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
#                                                   stride=1, padding=0, bias=False),
#                                         nn.Sigmoid())

#     def forward(self, x):
#         bahs, chs, _, _ = x.size()
 
#         spa_se = self.spatial_se(x)
#         return spa_se



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
#                            RecolSemLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.classifier_bin = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels // 4, 2, 1)
#         )
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

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_sem_0 = self.PyramidPooling(x1)
#         out1_sem_rep = self.representation(out1_sem_0)
#         out2_sem_0 = self.PyramidPooling(x2)
#         out2_sem_rep = self.representation(out2_sem_0)

#         # out_bin0 = torch.abs(x1 - x2)
#         # out_bin1 = self.PyramidPooling(out_bin0)
#         # out_bin_rep = self.representation(out_bin1)

#         # return out1_scd, out2_scd, out1_scd_rep, out2_scd_rep, out_bin_rep
#         return out1_scd, out2_scd, out1_sem_rep, out2_sem_rep


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
#                            RecolScdLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         # self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.classifier_scd = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels // 4, 7, 1)
#         )
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

#         out1_scd_0 = self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1))
#         out2_scd_0 = self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1))
#         out1_scd_1 = self.PyramidPooling(out1_scd_0)
#         out2_scd_1 = self.PyramidPooling(out2_scd_0)
#         out1_scd_pred = self.classifier_scd(out1_scd_1)
#         out2_scd_pred = self.classifier_scd(out2_scd_1)
#         out1_scd_rep = self.representation(out1_scd_1)
#         out2_scd_rep = self.representation(out2_scd_1)

#         return out1_scd_pred, out2_scd_pred, out1_scd_rep, out2_scd_rep

        
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
#                            RecolScdLoss+RecolBinLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling_sem = PyramidPooling(in_channels)
#         self.PyramidPooling_bin = PyramidPooling(in_channels)
#         self.representation_sem = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels, lightweight),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels, 256, 1)
#         )
#         self.representation_bin = nn.Sequential(
#         conv3x3(in_channels + 4 * int(in_channels / 4), in_channels, lightweight),
#         nn.BatchNorm2d(in_channels),
#         nn.ReLU(True),
#         # nn.Dropout(0.1, False),
#         nn.Conv2d(in_channels, 256, 1)
#         )



#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_sem_0 = self.PyramidPooling_sem(x1)
#         out1_sem_rep = self.representation_sem(out1_sem_0)
#         out2_sem_0 = self.PyramidPooling_sem(x2)
#         out2_sem_rep = self.representation_sem(out2_sem_0)

#         out_bin0 = torch.abs(x1 - x2)
#         out_bin1 = self.PyramidPooling_bin(out_bin0)
#         out_bin_rep = self.representation_bin(out_bin1)

#         return out1_scd, out2_scd, out1_sem_rep, out2_sem_rep, out_bin_rep


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
#                           ScdLoss + Spacial Attention
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead
# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600
#         self.sa = Spatial_Attention_Module()
#         self.head_scd = FCNHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out_bin = torch.abs(x1 - x2)
#         out_bin_sa = self.sa(out_bin)
#         out1_scd = self.head_scd(x1*out_bin_sa)
#         out2_scd = self.head_scd(x2*out_bin_sa)

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out1_scd, out2_scd


#------------Spatial attention module in CBAM------------------------------------
# class Spatial_Attention_Module(nn.Module):
#     def __init__(self):
#         super(Spatial_Attention_Module, self).__init__()
#         self.avg_pooling = torch.mean
#         self.max_pooling = torch.max
#         # In order to keep the size of the front and rear images consistent
#         # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
#         # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
#         # assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
#         # self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
#         #                       bias = False)
#         self.conv = nn.Conv2d(2, 1, kernel_size = (3, 3), stride = (1, 1), padding = ((3 - 1) // 2, (3 - 1) // 2),
#                               bias = False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # compress the C channel to 1 and keep the dimensions
#         avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
#         max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
#         v = self.conv(torch.cat((max_x, avg_x), dim = 1))
#         v = self.sigmoid(v)
#         return v


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


# #-----------------------------------------------------------------------------------
# #                           ScdLoss + Spacial Attention+Channel Attention1
# #-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600
#         self.sa = Spatial_Attention_Module()
#         self.ca = Channel_Attention_Module_FC(in_channels, 10)
#         self.head_scd = FCNHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out_bin = torch.abs(x1 - x2)
#         out_bin_sa = self.sa(out_bin)
#         x1_ca = self.ca(x1)
#         scd1_1 = out_bin_sa * x1
#         scd1_2 = x1_ca * scd1_1
#         x2_ca = self.ca(x2)
#         scd2_1 = out_bin_sa * x2
#         scd2_2 = x2_ca * scd2_1
        
#         out1_scd = self.head_scd(scd1_2)
#         out2_scd = self.head_scd(scd2_2)

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out1_scd, out2_scd


# #------------Spatial attention module in CBAM------------------------------------
# class Spatial_Attention_Module(nn.Module):
#     def __init__(self):
#         super(Spatial_Attention_Module, self).__init__()
#         self.avg_pooling = torch.mean
#         self.max_pooling = torch.max
#         self.conv = nn.Conv2d(2, 1, kernel_size = (1, 1), stride = (1, 1), padding = ((1 - 1) // 2, (1 - 1) // 2),
#                               bias = False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # compress the C channel to 1 and keep the dimensions
#         avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
#         max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
#         v = self.conv(torch.cat((max_x, avg_x), dim = 1))
#         v = self.sigmoid(v)
#         return v


# class Channel_Attention_Module_FC(nn.Module):
#     def __init__(self, channels, ratio):
#         super(Channel_Attention_Module_FC, self).__init__()
#         self.avg_pooling = nn.AdaptiveAvgPool2d(1)
#         self.max_pooling = nn.AdaptiveMaxPool2d(1)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
#             nn.ReLU(),
#             nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, h, w = x.shape
#         avg_x = self.avg_pooling(x).view(b, c)
#         max_x = self.max_pooling(x).view(b, c)
#         v = self.fc_layers(avg_x) + self.fc_layers(max_x)
#         v = self.sigmoid(v).view(b, c, 1, 1)
#         return v



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
#                           ScdLoss+BinLoss+share
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = PSPHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 
#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out_bin.squeeze(1), out1_scd, out2_scd

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




# #-----------------------------------------------------------------------------------
# #                           ScdLoss+BinLoss
# #-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_seg = PSPHead(in_channels, nclass-1, lightweight) 
#         self.head_bin = PSPHead(in_channels, 1, lightweight) 
#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

#     def base_forward(self, x1, x2):
#         b, c, h, w = x1.shape

#         x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
#         x2 = self.backbone.base_forward(x2)[-1]

#         out_bin = torch.abs(x1 - x2)
#         out_bin = self.head_bin(out_bin) #[2,1,128,128]
#         out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
#         out_bin = torch.sigmoid(out_bin)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         return out_bin.squeeze(1), out1_scd, out2_scd

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
#                           ScdLoss
#-----------------------------------------------------------------------------------
from models.block.conv import conv3x3
from models.sseg.base import BaseNet
from models.sseg.fcn import FCNHead

import torch
from torch import nn
import torch.nn.functional as F


class PSPNet(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(PSPNet, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1] # 600
        self.head_scd = PSPHead(in_channels, nclass, lightweight) 
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
        x2 = self.backbone.base_forward(x2)[-1]

        out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
        out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

        out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
        out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

        return out1_scd, out2_scd


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(PyramidPooling(in_channels),
                                   conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)




#-----------------------------------------------------------------------------------
#                            RecolBinLoss+ScdLoss
#-----------------------------------------------------------------------------------
# from models.block.conv import conv3x3
# from models.sseg.base import BaseNet
# from models.sseg.fcn import FCNHead

# import torch
# from torch import nn
# import torch.nn.functional as F


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1] # 600

#         self.head_scd = PSPHead(in_channels, nclass, lightweight) 
#         self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
#         self.PyramidPooling = PyramidPooling(in_channels)
#         self.classifier_bin = nn.Sequential(
#             conv3x3(in_channels + 4 * int(in_channels / 4), in_channels // 4, lightweight),
#             nn.BatchNorm2d(in_channels // 4),
#             nn.ReLU(True),
#             # nn.Dropout(0.1, False),
#             nn.Conv2d(in_channels // 4, 2, 1)
#         )
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

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out_bin0 = torch.abs(x1 - x2)
#         out_bin1 = self.PyramidPooling(out_bin0)
#         out_bin_rep = self.representation(out_bin1)

#         return out1_scd, out2_scd, out_bin_rep


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


# class PSPNet(BaseNet):
#     def __init__(self, backbone, pretrained, nclass, lightweight):
#         super(PSPNet, self).__init__(backbone, pretrained)

#         in_channels = self.backbone.channels[-1]
#         self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
#         self.head_sem =FCNHead(in_channels, nclass-1, lightweight)
#         self.head_scd = PSPHead(in_channels, nclass, lightweight)
#         self.head_sem_rep = nn.Sequential(conv3x3(in_channels, in_channels // 4, lightweight),
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

#         out1_sem_rep = self.head_sem_rep(x1)
#         out2_sem_rep = self.head_sem_rep(x2)

#         out1_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1)))
#         out2_scd = self.head_scd(self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1)))

#         out1_scd = F.interpolate(out1_scd, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_scd = F.interpolate(out2_scd, size=(h, w), mode='bilinear', align_corners=False)

#         out1_sem_rep = F.interpolate(out1_sem_rep, size=(104, 104), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
#         out2_sem_rep = F.interpolate(out2_sem_rep, size=(104, 104), mode='bilinear', align_corners=False)


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