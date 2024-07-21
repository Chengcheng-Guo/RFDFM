# -*- coding: UTF-8 -*-
# RFDFM source code
# May-22-2024
import argparse
import os
import time
import datetime


import torch
from monai.losses import DiceCELoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
from about_log import logger
from common import create_dir_if_not_exists
from RFDFM import RFDFM
from utils import custom_data, auxiliary_data, custom_data_val

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, loss_function, device, epoch, iters_verbose=10,
                    auxiliary_train_loader=None, auxiliary_train_iter=None):
    binary_loss = BCEWithLogitsLoss()
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        epoch_loss = 0
        step = 0
        total_step = len(train_loader)
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            if auxiliary_train_loader is not None:
                try:
                    nxt = next(auxiliary_train_iter)
                except:
                    auxiliary_train_iter = iter(auxiliary_train_loader)
                    nxt = next(auxiliary_train_iter)
                optimizer.zero_grad()
                pred = model.forward_seg(nxt["image"].to(device))
                dice_loss = loss_function[0](pred, nxt["mask"].to(device)) * 0.8
                dice_loss.backward()
                optimizer.step()
                del pred, nxt
                dice_loss = dice_loss.item()
            else:
                dice_loss = torch.tensor(0.0, device=device)
                # optimizer.zero_grad()
                
                
            optimizer.zero_grad()
            inputs, masks, labels = (batch_data["image"].to(device),
                                     batch_data["mask"].to(device),
                                     batch_data["label"].to(device))
            seg, clf, rad, final = model(inputs)
            seg_loss_function, clf_loss_function = loss_function
            seg_loss = seg_loss_function(seg, masks)
            clf_clf_loss = clf_loss_function(clf, labels)
            clf_rad_loss = binary_loss(rad.reshape(-1), labels.type(rad.dtype))
            # print("====", type(clf_rad_loss), type(clf_clf_loss), clf, final, labels, "====")
            clf_final_loss = clf_loss_function(final, labels)
            clf_loss = 0.1 * clf_clf_loss + \
                       0.1 * clf_rad_loss + clf_final_loss
            loss = seg_loss + clf_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            if step % iters_verbose == 0:
                logger.info(f"TRAIN EPOCH: {epoch}, step:{step}/{total_step}, train_loss: {loss.item():.4f}, "
                      f"seg_train_loss: {seg_loss.item():.4f}, DICE: {(1.0 - seg_loss.item()):.4f}, clf_train_loss: {clf_loss.item():.4f}, "
                      f"clf_clf_loss: {clf_clf_loss}, clf_rad_loss: {clf_rad_loss}, clf_final_loss: {clf_final_loss}, "
                      f"final_auc: {(0.0 if labels.max() == labels.min() else roc_auc_score(labels.detach().cpu().numpy(), torch.softmax(final.detach(), -1)[:,1].cpu().numpy())):.6f}, "
                      f"final_acc: {torch.eq(torch.argmax(final.detach(), dim=-1), labels).float().mean().item():.6f}, "
                      f"lr: {lr_scheduler.get_last_lr()[0]:.6f}, "
                      f"step time: {(time.time() - step_start):.4f}, "
                      f"auxiliary dice: {dice_loss:.4f}")
        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}\n\n")
        return epoch_loss
def diceCoeff(pred, gt, smooth=1e-5):
  r""" computational formula：
    dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
  """

  pred_flat = torch.flatten(pred, start_dim=1, end_dim=-1)
  gt_flat = torch.flatten(gt, start_dim=1, end_dim=-1)

  intersection = torch.sum(pred_flat * gt_flat, dim=1)
  unionset = pred_flat.sum(1) + gt_flat.sum(1)
  loss = (2 * intersection + smooth) / (unionset + smooth)

  return loss

def evaluate(val_loader, model, device, epoch:int, clf_thres=None, rad_thres=None, final_thres=None):
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
        logger.info(f"VAL EPOCH: {epoch}, DICE: {dice}")


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
        logger.info(f"VAL EPOCH: {epoch}, Clf ACC: {clf_acc:.5f}, Clf AUC: {clf_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
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
        logger.info(f"VAL EPOCH: {epoch}, Rad ACC: {rad_acc:.5f}, Rad AUC: {rad_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
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
        logger.info(f"VAL EPOCH: {epoch}, Final ACC: {final_acc:.5f}, Final AUC: {final_auc:.5f}, ppv: {ppv:.5f}, npv: {npv:.5f}")
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

    train_ds = custom_data(args.data_dir, train_files, aug_prob=(list(map(float, args.aug_prob)) if args.aug_prob is not None else args.aug_prob))
    val_ds = custom_data_val(args.data_dir,val_files)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.j, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.j, shuffle=False)
    logger.info(f"Train data size {len(train_ds)}, Val data size {len(val_ds)}.")

    if args.auxiliary_train:
        auxiliary_train_ds = auxiliary_data(args.data_dir, train_files)
        auxiliary_train_loader = torch.utils.data.DataLoader(auxiliary_train_ds, batch_size=args.auxiliary_batch_size, num_workers=args.j // 2, shuffle=True)
        auxiliary_train_iter = iter(auxiliary_train_loader)
    else:
        auxiliary_train_loader = None
        auxiliary_train_iter = None

    # model, optimizer, loss
    model = RFDFM(spatial_dims=3, in_channels=1, out_channels=64, strides=2, channels=[32, 64], num_classes=2)
    model = model.to(device)

    seg_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    clf_loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-5)

    
    if args.model_path is not None and os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        # load best core weights
        model.load_state_dict(state_dict["state_dict"])
        optimizer.load_state_dict(state_dict["optimizer"])
        last_epoch = state_dict["epoch"]
    else:
        last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.epochs * len(train_ds) // args.batch_size,
                                                              last_epoch=(-1 if last_epoch == -1 else (last_epoch + 1) * len(train_ds) // args.batch_size))

    save_dir = create_dir_if_not_exists(args.save_dir, add_date=True)
    tstmp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = create_dir_if_not_exists(os.path.join(save_dir, args.model_name + "_" + tstmp))
    logger.info(f"saving to {args.save_dir}")
    best_metric = 0
    best_metric_epoch = 0
    # train
    for epoch in range(last_epoch + 1, args.epochs):
        # 1 Epoch
        train_one_epoch(train_loader, model, optimizer, lr_scheduler, (seg_loss_function, clf_loss_function), device,
                        epoch + 1, args.iters_verbose, auxiliary_train_loader=auxiliary_train_loader,
                        auxiliary_train_iter=auxiliary_train_iter)
        # validation
        if epoch % args.val_interval == 0:
            final_acc, final_auc, final_thres, dice, clf_thres, rad_thres = evaluate(val_loader, model, device, epoch + 1)

            if final_auc > best_metric:
                best_metric = final_auc
                best_metric_epoch = epoch + 1
                torch.save({'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clf_thres': clf_thres, 'rad_thres': rad_thres, 'final_thres': final_thres},
                os.path.join(save_dir, f"{args.model_name}_epo{epoch}_{best_metric:.4f}_{final_thres}.pth"))

                logger.info("----------")
                logger.info("saved new best metric model")
                logger.info("----------")

            else:
                torch.save({'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clf_thres': clf_thres, 'rad_thres': rad_thres, 'final_thres': final_thres},
                os.path.join(save_dir, f"{args.model_name}_epo{epoch}_{final_auc:.4f}_{final_thres}.pth"))
            logger.info(f"current epoch: {epoch + 1} current auc: {final_auc:.4f}"
                  f"\nbest auc: {best_metric:.4f} at epoch: {best_metric_epoch}")



DATA_ROOT = r'C:\Users\onekey\Project\OnekeyDS\CT\crop_3d'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', type=str, default=DATA_ROOT, help='Training dataset')
    parser.add_argument('--valid', type=str, default=None, help='Validation dataset')
    parser.add_argument("--auxiliary_train", action='store_true', default=False)
    parser.add_argument('--auxiliary_batch_size', default=4, type=int)
    parser.add_argument('--roi_size', nargs='*', default=[48, 48, 16], type=int, help='ROI size')
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--model_name', default='RFDFM', help='Model name')
    parser.add_argument('--model_path', default='', type=str, help='checkpoint path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to be used!')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--iters_verbose', default=1, type=int, help='print frequency')
    parser.add_argument('--val_interval', default=1, type=int, help='print frequency')
    parser.add_argument('--cached_ratio', default=None, type=float, help='cached ratio')
    parser.add_argument('--data_dir', default="/teams/Lung_seg_1695727983/guochengcheng/32_slice/ct/", type=str, help='data dir')
    parser.add_argument('--aug_prob', nargs='*', default=None, help='augumentationn prob')
    main(parser.parse_args())
