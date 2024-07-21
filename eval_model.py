# -*- coding: UTF-8 -*-
# RFDFM source code
# May-22-2024

import argparse
import os

import torch
from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
from about_log import logger
from RFDFM import RFDFM
from utils import custom_data, custom_data_val
from run_model import diceCoeff

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evaluate(val_loader, model, device, clf_thres=None, rad_thres=None, final_thres=None):
    model.eval()
    clf_acc = 0
    rad_acc = 0
    final_acc = 0


    dice_list = []
    clf_list = []
    rad_list = []
    final_list = []
    label_list = []
    data_num = 0.0

    with torch.no_grad():
        for val_data in val_loader:
            inputs, masks, labels = (val_data["image"].to(device),
                                     val_data["mask"].to(device),
                                     val_data["label"])
            seg, clf, rad, final = model(inputs)
            dice_list += [diceCoeff(torch.softmax(seg, axis=1)[:,[1]], masks)]
            clf_list += [torch.softmax(clf, -1)[:,1].cpu().numpy()]
            # clf_acc = clf_acc + torch.eq(torch.argmax(clf, dim=-1).cpu(), labels).sum().item()

            rad_list += [torch.sigmoid(rad).reshape(-1).cpu().numpy()]
            # rad_acc = rad_acc + torch.eq(torch.argmax(rad, dim=-1).cpu(), labels).sum().item()

            final_list += [torch.softmax(final, -1)[:,1].cpu().numpy()]
            # final_acc = final_acc + torch.eq(torch.argmax(final, dim=-1).cpu(), labels).sum().item()

            label_list += [labels]

            data_num += float(inputs.shape[0])

        dice = torch.mean(torch.cat(dice_list, axis=-1)).item()
        logger.info(f"VAL DICE: {dice}")


        clf_list = np.concatenate(clf_list, axis=-1)
        rad_list = np.concatenate(rad_list, axis=-1)
        final_list = np.concatenate(final_list, axis=-1)

        label_list = np.concatenate(label_list, axis=-1)

        clf_auc = roc_auc_score(label_list, clf_list)
        rad_auc = roc_auc_score(label_list, rad_list)
        final_auc = roc_auc_score(label_list, final_list)

        if clf_thres is None:
            thresholds = clf_list.copy()
            preds = (clf_list.copy().reshape(1, -1) >= clf_list.copy().reshape(-1, 1))
            acc = (preds == label_list.reshape(1, -1)).sum(axis = 1)
            clf_thres = thresholds[np.argmax(acc)]
        clf_acc = ((clf_list >= clf_thres).astype(np.int32) == label_list).mean()
        tp = np.logical_and(clf_list >= clf_thres, label_list).sum()
        fp = np.logical_and(clf_list >= clf_thres, 1 - label_list).sum()
        tn = np.logical_and(clf_list < clf_thres, 1 - label_list).sum()
        fn = np.logical_and(clf_list < clf_thres, label_list).sum()
        
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        sense = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1 = 2 * ppv * sense / (ppv + sense + 1e-8)
        logger.info(f"VAL Clf ACC: {clf_acc:.5f}, Clf AUC: {clf_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
        logger.info(f"           sense: {sense:.5f}, spec: {spec:.5f}, f1: {f1:.5f}, npv: {npv:.5f}")
        logger.info(f"           tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, threshold: {clf_thres}")
        logger.info("----------------------------")

        if rad_thres is None:
            thresholds = rad_list.copy()
            preds = (rad_list.copy().reshape(1, -1) >= rad_list.copy().reshape(-1, 1))
            acc = (preds == label_list.reshape(1, -1)).sum(axis = 1)
            rad_thres = thresholds[np.argmax(acc)]
        rad_acc = ((rad_list >= rad_thres).astype(np.int32) == label_list).mean()
        tp = np.logical_and(rad_list >= rad_thres, label_list).sum()
        fp = np.logical_and(rad_list >= rad_thres, 1 - label_list).sum()
        tn = np.logical_and(rad_list < rad_thres, 1 - label_list).sum()
        fn = np.logical_and(rad_list < rad_thres, label_list).sum()
        
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        sense = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1 = 2 * ppv * sense / (ppv + sense + 1e-8)
        logger.info(f"VAL Rad ACC: {rad_acc:.5f}, Rad AUC: {rad_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
        logger.info(f"           sense: {sense:.5f}, spec: {spec:.5f}, f1: {f1:.5f}, npv: {npv:.5f}")
        logger.info(f"           tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, threshold: {rad_thres}")
        logger.info("----------------------------")

        if final_thres is None:
            thresholds = final_list.copy()
            preds = (final_list.copy().reshape(1, -1) >= final_list.copy().reshape(-1, 1))
            acc = (preds == label_list.reshape(1, -1)).sum(axis = 1)
            final_thres = thresholds[np.argmax(acc)]
        final_acc = ((final_list >= final_thres).astype(np.int32) == label_list).mean()
        tp = np.logical_and(final_list >= final_thres, label_list).sum()
        fp = np.logical_and(final_list >= final_thres, 1 - label_list).sum()
        tn = np.logical_and(final_list < final_thres, 1 - label_list).sum()
        fn = np.logical_and(final_list < final_thres, label_list).sum()
        
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        sense = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1 = 2 * ppv * sense / (ppv + sense + 1e-8)
        logger.info(f"VAL Final ACC: {final_acc:.5f}, Final AUC: {final_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
        logger.info(f"           sense: {sense:.5f}, spec: {spec:.5f}, f1: {f1:.5f}, npv: {npv:.5f}")
        logger.info(f"           tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, threshold: {final_thres}")
        logger.info("----------------------------")
        logger.info(f"VAL num: {data_num}")
        return final_acc, final_auc, final_thres, dice, clf_thres, rad_thres


def main(args, trans=None, ):
    if args.valid is None:
        train_files, val_files = args.train
    else:
        train_files = args.train
        val_files = args.valid

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_ds = custom_data(args.data_dir, train_files)
    val_ds = custom_data_val(args.data_dir,val_files)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.j, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.j, shuffle=False)
    logger.info(f"Train data size {len(train_ds)}, Val data size {len(val_ds)}.")

    model = RFDFM(spatial_dims=3, in_channels=1, out_channels=64, strides=2, channels=[32, 64], num_classes=2)
    model = model.to(device)
    # Save the best training parameters.
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    # load best core weights
    model.load_state_dict(state_dict)

    evaluate(val_loader, model, device, clf_thres=ckpt["clf_thres"], rad_thres=ckpt["rad_thres"], final_thres=ckpt["final_thres"])


DATA_ROOT = r'C:\Users\onekey\Project\OnekeyDS\CT\crop_3d'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', type=str, default=DATA_ROOT, help='Training dataset')
    parser.add_argument('--valid', type=str, default=None, help='Validation dataset')
    parser.add_argument('--model_path', default='./20221204/RFDFM/RFDFM.pth', help='ROI size')
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--model_name', default='RFDFM', help='Model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to be used!')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cached_ratio', default=None, type=float, help='cached ratio')
    parser.add_argument('--data_dir', default="/teams/Lung_seg_1695727983/guochengcheng/32_slice/ct/", type=str, help='data dir')
    main(parser.parse_args())
