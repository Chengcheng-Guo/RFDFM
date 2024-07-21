# -*- coding: UTF-8 -*-
# RFDFM source code
# May-22-2024

from functools import partial
from typing import Callable, Union
from typing import Sequence

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import get_pool_layer
from monai.networks.layers.factories import Conv, Norm

__all__ = ["RFDFM"]

import radiomics
import logging
from radiomics import featureextractor

radiomics.logger.setLevel(logging.ERROR)
from torch.nn import CrossEntropyLoss
from multiprocessing import Pool



def get_up_layer(in_channels, out_channels, channels, strides, kernel_size, dropout=0):
    trans_conv1 = Convolution(3, in_channels, channels[-1],
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    trans_conv2 = Convolution(3, channels[-1], channels[-2],
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    trans_conv3 = Convolution(3, channels[-2], out_channels,
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    return nn.Sequential(trans_conv1, trans_conv2, trans_conv3)


def get_radiomics_feature_core(img_msk, i, extractor, num_features):
    img = sitk.GetImageFromArray(img_msk[i, 0])
    msk = sitk.GetImageFromArray(img_msk[i, 1])
    try:
        featureVector = extractor.execute(img, msk, label=1)
        return [float(featureVector[featureName]) if not np.isnan(featureVector[featureName]) else 0
                for featureName in featureVector.keys() if 'diagnostics' not in featureName]
    except:
        print(f"padding with 0...")
        return [0] * num_features


def get_radiomics_feature(img_: torch.Tensor, msk_: torch.Tensor, extractor, num_features) -> torch.Tensor:
    img_msk = torch.cat([img_, msk_], axis=1).detach().cpu().numpy()
    bs = img_msk.shape[0]
    with Pool(8) as p:
        features = p.starmap(get_radiomics_feature_core,
                             zip([img_msk] * bs, range(bs), [extractor] * bs, [num_features] * bs))

    features = torch.from_numpy(np.array(features)).float().to('cuda:0')
    return features


def get_radiomics_feature_old(img_: torch.Tensor, msk_: torch.Tensor, extractor, num_features) -> torch.Tensor:
    features = []
    img_ = torch.squeeze(img_, dim=1)
    msk_ = torch.squeeze(msk_, dim=1)
    for img, msk in zip(img_, msk_):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            img = sitk.GetImageFromArray(img)
        if isinstance(msk, torch.Tensor):
            msk = msk.detach().cpu().numpy()
            msk = sitk.GetImageFromArray(msk)
        try:
            featureVector = extractor.execute(img, msk, label=1)
            features.append([float(featureVector[featureName]) if not np.isnan(featureVector[featureName]) else 0
                             for featureName in featureVector.keys() if 'diagnostics' not in featureName])
        except:
            print(f"padding with 0...")
            features.append([0] * num_features)
    features = torch.from_numpy(np.array(features)).float().to('cuda:0')
    return features


class RFDFM(nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_rad_features: int = 107,
            kernel_size: Union[Sequence[int], int] = 3,
            **kwargs) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 3]


        self.seg_conv1 = nn.Conv3d(1, 8, kernel_size=5, stride=1, bias=False, padding="same")
        self.seg_bn1 = nn.BatchNorm3d(8)
        self.seg_conv2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, bias=False, padding="same")
        self.seg_bn2 = nn.BatchNorm3d(8)

        self.seg_conv3 = nn.Conv3d(8, 32, kernel_size=5, stride=2, bias=False, padding=2)
        self.seg_bn3 = nn.BatchNorm3d(32)
        self.seg_conv4 = nn.Conv3d(32, 32, kernel_size=1, stride=1, bias=False, padding="same")
        self.seg_bn4 = nn.BatchNorm3d(32)
        self.seg_conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, bias=False, padding="same")
        self.seg_bn5 = nn.BatchNorm3d(32)

        self.seg_conv6 = nn.Conv3d(40, 64, kernel_size=5, stride=2, bias=False, padding=2)
        self.seg_bn6 = nn.BatchNorm3d(64)
        self.seg_conv7 = nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=False, padding="same")
        self.seg_bn7 = nn.BatchNorm3d(64)
        self.seg_conv8 = nn.Conv3d(64, 64, kernel_size=3, stride=1, bias=False, padding="same")
        self.seg_bn8 = nn.BatchNorm3d(64)

        self.seg_conv9 = nn.Conv3d(72, 128, kernel_size=5, stride=2, bias=False, padding=2)
        self.seg_bn9 = nn.BatchNorm3d(128)
        self.seg_conv10 = nn.Conv3d(128, 128, kernel_size=1, stride=1, bias=False, padding="same")
        self.seg_bn10 = nn.BatchNorm3d(128)
        self.seg_conv11 = nn.Conv3d(128, 128, kernel_size=3, stride=1, bias=False, padding="same")
        self.seg_bn11 = nn.BatchNorm3d(128)


        self.clf_conv1 = nn.Conv3d(1, 8, kernel_size=5, stride=1, bias=False, padding="same")
        self.clf_bn1 = nn.BatchNorm3d(8)
        self.clf_conv2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, bias=False, padding="same")
        self.clf_bn2 = nn.BatchNorm3d(8)

        self.clf_conv3 = nn.Conv3d(8, 32, kernel_size=5, stride=2, bias=False, padding=2)
        self.clf_bn3 = nn.BatchNorm3d(32)
        self.clf_conv4 = nn.Conv3d(32, 32, kernel_size=1, stride=1, bias=False, padding="same")
        self.clf_bn4 = nn.BatchNorm3d(32)
        self.clf_conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, bias=False, padding="same")
        self.clf_bn5 = nn.BatchNorm3d(32)

        self.clf_conv6 = nn.Conv3d(40, 64, kernel_size=5, stride=2, bias=False, padding=2)
        self.clf_bn6 = nn.BatchNorm3d(64)
        self.clf_conv7 = nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=False, padding="same")
        self.clf_bn7 = nn.BatchNorm3d(64)
        self.clf_conv8 = nn.Conv3d(64, 64, kernel_size=3, stride=1, bias=False, padding="same")
        self.clf_bn8 = nn.BatchNorm3d(64)

        self.clf_conv9 = nn.Conv3d(72, 128, kernel_size=5, stride=2, bias=False, padding=2)
        self.clf_bn9 = nn.BatchNorm3d(128)
        self.clf_conv10 = nn.Conv3d(128, 128, kernel_size=1, stride=1, bias=False, padding="same")
        self.clf_bn10 = nn.BatchNorm3d(128)
        self.clf_conv11 = nn.Conv3d(128, 128, kernel_size=3, stride=1, bias=False, padding="same")
        self.clf_bn11 = nn.BatchNorm3d(128)

        self.num_cnn_channel = 136

        # Seg decoder
        self.seg_decoder = get_up_layer(self.num_cnn_channel, 2, channels=[64, 32], strides=2, kernel_size=kernel_size)
        # Clf decoder
        self.clf_decoder = get_pool_layer(("adaptiveavg", {"output_size": (1, 1, 1)}), spatial_dims=3)
        self.clf_task = torch.nn.Linear(self.num_cnn_channel, num_classes, bias=False)
        
        self.num_rad_features = num_rad_features
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

        # FSM
        self.fsm_act = torch.nn.SELU(inplace=True)
        self.fsm_bn = torch.nn.BatchNorm1d(num_rad_features)
        self.fsm1 = torch.nn.Linear(num_rad_features + self.num_cnn_channel, 128, bias=False)
        self.fsm2 = torch.nn.Linear(128, num_rad_features, bias=False)

        # DeepFM
        self.lr = torch.nn.Linear(num_rad_features, 1, bias=True)
        self.fm = nn.Parameter(torch.Tensor(32, num_rad_features))
        nn.init.xavier_uniform_(self.fm)
        self.deep1 = torch.nn.Linear(num_rad_features, 64, bias=False)
        self.deep2 = torch.nn.Linear(64, 32, bias=False)
        self.deep3 = torch.nn.Linear(32, 1, bias=False)

        # Rad task
        self.rad_task = torch.nn.Linear(num_rad_features, num_classes, bias=False)

        # Final task
        self.final_act = torch.nn.ReLU(inplace=True)
        num_concat_features = self.num_cnn_channel + num_rad_features
        self.fc1 = torch.nn.Linear(num_concat_features, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, 32, bias=False)
        self.final_task = torch.nn.Linear(32 + 1 + 1 + 32, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        # Initial Conv
        seg_out = torch.relu(self.seg_bn1(self.seg_conv1(x)))
        src_seg = torch.relu(self.seg_bn2(self.seg_conv2(seg_out))) # [b, 8, 64, 64, 64]
        ########
        clf_out = torch.relu(self.clf_bn1(self.clf_conv1(x)))
        src_clf = torch.relu(self.clf_bn2(self.clf_conv2(clf_out))) # [b, 8, 64, 64, 64]

        # Block 1
        seg_out = torch.relu(self.seg_bn3(self.seg_conv3(src_seg)))
        seg_out = torch.relu(self.seg_bn4(self.seg_conv4(seg_out)))
        seg_out = self.seg_conv5(seg_out)
        atten_out = torch.softmax(self.seg_bn5(seg_out), dim=1)
        seg_out = torch.relu(seg_out)
        ########
        clf_out = torch.relu(self.clf_bn3(self.clf_conv3(src_clf)))
        clf_out = torch.relu(self.clf_bn4(self.clf_conv4(clf_out)))
        clf_out = self.clf_conv5(clf_out)
        clf_out = torch.relu(clf_out + self.clf_bn5(clf_out) * atten_out)

        # Block 2
        src_seg = torch.nn.functional.avg_pool3d(src_seg, kernel_size=2)
        seg_out = torch.relu(self.seg_bn6(self.seg_conv6(torch.cat([seg_out, src_seg], dim=1))))
        seg_out = torch.relu(self.seg_bn7(self.seg_conv7(seg_out)))
        seg_out = self.seg_conv8(seg_out)
        atten_out = torch.softmax(self.seg_bn8(seg_out), dim=1)
        seg_out = torch.relu(seg_out)
        ########
        src_clf = torch.nn.functional.avg_pool3d(src_clf, kernel_size=2)
        clf_out = torch.relu(self.clf_bn6(self.clf_conv6(torch.cat([clf_out, src_clf], dim=1))))
        clf_out = torch.relu(self.clf_bn7(self.clf_conv7(clf_out)))
        clf_out = self.clf_conv8(clf_out)
        clf_out = torch.relu(clf_out + self.clf_bn8(clf_out) * atten_out)

        # Block 3
        src_seg = torch.nn.functional.avg_pool3d(src_seg, kernel_size=2)
        seg_out = torch.relu(self.seg_bn9(self.seg_conv9(torch.cat([seg_out, src_seg], dim=1))))
        seg_out = torch.relu(self.seg_bn10(self.seg_conv10(seg_out)))
        seg_out = self.seg_conv11(seg_out)
        atten_out = torch.softmax(self.seg_bn11(seg_out), dim=1)
        seg_out = torch.relu(seg_out)
        ########
        src_clf = torch.nn.functional.avg_pool3d(src_clf, kernel_size=2)
        clf_out = torch.relu(self.clf_bn9(self.clf_conv9(torch.cat([clf_out, src_clf], dim=1))))
        clf_out = torch.relu(self.clf_bn10(self.clf_conv10(clf_out)))
        clf_out = self.clf_conv11(clf_out)
        clf_out = torch.relu(clf_out + self.clf_bn11(clf_out) * atten_out)


        src_seg = torch.nn.functional.avg_pool3d(src_seg, kernel_size=2)
        src_clf = torch.nn.functional.avg_pool3d(src_clf, kernel_size=2)
        seg_out = torch.cat([seg_out, src_seg], dim=1)
        clf_out = torch.cat([clf_out, src_clf], dim=1)

        mask = self.seg_decoder(seg_out)
        seg_task = torch.argmax(mask, 1, keepdim=True)
        clf_out = self.clf_decoder(clf_out)
        clf_out = torch.reshape(clf_out, (-1, self.num_cnn_channel))

        # FSM
        rad_features = get_radiomics_feature(x, seg_task, self.extractor, self.num_rad_features)
        rad_features = self.fsm_bn(rad_features)
        fsm = self.fsm_act(self.fsm1(torch.cat([rad_features, clf_out.detach()], dim=1)))
        fsm = self.fsm_act(self.fsm2(fsm))  
        fsm = torch.softmax(fsm, dim=1)
        rad_features_weighted = fsm * rad_features

        # LR logit
        lr = self.lr(rad_features_weighted)

        # FM logit
        square_of_sum = torch.mm(rad_features_weighted, self.fm.T) * torch.mm(rad_features_weighted, self.fm.T)
        sum_of_square = torch.mm(rad_features_weighted * rad_features_weighted, self.fm.T * self.fm.T)
        fm = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=-1, keepdim=True)

        # Deep
        deep_vec = torch.relu(self.deep1(rad_features_weighted))
        deep_vec = torch.relu(self.deep2(deep_vec))
        deep = self.deep3(deep_vec)

        rad_task = lr + fm + deep

        # Classification
        clf_task = self.clf_task(clf_out)

        # Final task
        concat_features = torch.cat([clf_out, rad_features_weighted], dim=1)
        fc = self.final_act(self.fc1(concat_features))
        fc = self.final_act(self.fc2(fc))
        final_task = self.final_task(torch.cat([fc, lr, fm, deep_vec], dim=1))
        return mask, clf_task, rad_task, final_task


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        a = torch.rand((4, 1, 32, 32, 32)).cuda()
        m = torch.randint(0, 2, (4, 1, 32, 32, 32)).cuda()
        c = torch.randint(0, 2, (4,)).cuda()
        rfdfm = RFDFM(spatial_dims=3, in_channels=1, out_channels=64, strides=2, channels=[32, 64],
                      num_classes=2).cuda()
        seg, clf, rad, final = rfdfm(a)
        print(a.shape, m.shape, seg.shape, clf.shape, rad.shape, final.shape, c)
        seg_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        clf_loss_function = CrossEntropyLoss()
        seg_loss = seg_loss_function(seg, m)
        clf_loss = clf_loss_function(clf, c) + clf_loss_function(rad, c) + clf_loss_function(final, c)
        loss = seg_loss + clf_loss
        print(seg_loss + clf_loss, seg_loss, clf_loss)
        loss.backward()


