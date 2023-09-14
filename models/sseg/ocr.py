
#-----------------------------------------------------------------------------------
#                            Scd
#-----------------------------------------------------------------------------------
from models.block.conv import conv3x3
from models.sseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


# class ModuleHelper:

#     @staticmethod
#     def BNReLU(num_features, bn_type=None, **kwargs):
#         return nn.Sequential(
#             BatchNorm2d(num_features, **kwargs),
#             nn.ReLU()
#         )

#     @staticmethod
#     def BatchNorm2d(*args, **kwargs):
#         return BatchNorm2d


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context



class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context
    

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
    


class OCR(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(OCR, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1]
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)

        self.ocr_mid_channels = 512
        self.ocr_key_channels = 256

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, self.ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ocr_mid_channels),
            nn.ReLU(True),
        )

        self.ocr_gather_head = SpatialGather_Module(nclass)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.ocr_mid_channels,
                                                 key_channels=self.ocr_key_channels,
                                                 out_channels=self.ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            self.ocr_mid_channels, nclass, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, nclass,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        



    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self.backbone.base_forward(x1)[-1] # x1=[2,600,128,128]
        x2 = self.backbone.base_forward(x2)[-1]


        x1 = self.conv1x1(torch.cat((torch.abs(x1 - x2), x1), 1))
        x2 = self.conv1x1(torch.cat((torch.abs(x1 - x2), x2), 1))

        # out_aux_seg1 = []
        # ocr
        out_aux1 = self.aux_head(x1)
        # compute contrast feature
        x1 = self.conv3x3_ocr(x1)

        context1 = self.ocr_gather_head(x1, out_aux1)
        x1 = self.ocr_distri_head(x1, context1)

        out1 = self.cls_head(x1)

        # out_aux_seg1.append(out_aux1)
        # out_aux_seg1.append(out1)


        # out_aux_seg2 = []
        # ocr
        out_aux2 = self.aux_head(x2)
        # compute contrast feature
        x2 = self.conv3x3_ocr(x2)

        context2 = self.ocr_gather_head(x2, out_aux2)
        x2 = self.ocr_distri_head(x2, context2)

        out2 = self.cls_head(x2)

        # out_aux_seg2.append(out_aux2)
        # out_aux_seg2.append(out2)

        out1_scd = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False) # out1=[2,7,512,512]
        out2_scd = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

        return out1_scd, out2_scd



  
