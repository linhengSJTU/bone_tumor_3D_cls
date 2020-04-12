import os
import time
import glob
import cv2
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
from PIL import Image
# from apex import amp
from .tools import *
from .config import *
from torchvision import transforms
from data.custom_dataset import CustomDataset
from data.custom_collate import CustomCollate

cls_criterion = nn.BCELoss()
# cls_criterion = nn.CrossEntropyLoss()
ex_criterion = nn.MSELoss(reduction='none')


def prepare_net(config, model, _use='train'):
    if _use == 'train':
        if config['optim'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), config['lr'],
                                         weight_decay=config['weight_decay'], eps=1e-5)
        if config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            config['lr'], weight_decay=config['weight_decay'])
        elif config['optim'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

        folds = [fold for fold in range(1, config['n_fold'] + 1) if config['valid_fold'] != fold]
        train_dataset = CustomDataset('train', config['rescale_size'], config['FoldFile'], folds)
        config['train_length'] = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True,
                                                   num_workers=config['num_workers'], collate_fn=CustomCollate)

        valid_dataset = CustomDataset('valid', config['rescale_size'], config['FoldFile'], [config['valid_fold']])
        config['valid_length'] = len(valid_dataset)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batchsize'], shuffle=False,
                                                   num_workers=config['num_workers'], collate_fn=CustomCollate)

        return optimizer, train_loader, valid_loader

    elif _use == 'valid':
        valid_dataset = CustomDataset('valid', config['rescale_size'], config['FoldFile'], [config['valid_fold']])
        config['valid_length'] = len(valid_dataset)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                                   num_workers=config['num_workers'], collate_fn=CustomCollate)

        return valid_loader

    elif _use == 'infer':
        infer_dataset = CustomDataset('infer', config['rescale_size'], config['FoldFile'], None)
        config['infer_length'] = len(infer_dataset)
        infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=False,
                                                   num_workers=config['num_workers'], collate_fn=CustomCollate)

        return infer_loader

    else:
        raise NameError


def train_net(optimizer, train_loader, valid_loader, model, config):
    best_auc = 0
    best_m_f1 = 0
    best_fix_f1 = 0
    best_index = [best_auc, best_m_f1, best_fix_f1]

    if config['lr_decay'] == None:
        lr_decay = 0.1
    else:
        lr_decay = config['lr_decay']
    for epoch in range(3, config['num_epoch'] + 1):
        adjust_learning_rate(optimizer, epoch - 1, config['num_epoch'], config['lr'], config['lr_decay_freq'], lr_decay)

        train(train_loader, model, optimizer, epoch, config)

        if epoch % config['valid_freq'] == 0:
            best_index = valid_net(valid_loader, model, config, best_index, epoch)
            # logging.info('Valid-Cls: Best ACC update to: {:.4f}'.format(best_index[0]))
            # logging.info('Valid-Cls: Best AUC update to: {:.4f}'.format(best_index[1]))
            # logging.info('Valid-Cls: Best F1  update to: {:.4f}'.format(best_index[2]))
            # logging.info('Valid-Cls: Best SEN update to: {:.4f}'.format(best_index[3]))
            # logging.info('Valid-Cls: Best SPE update to: {:.4f}'.format(best_index[4]))


def valid_net(valid_loader, model, config, best_index, epoch):
    m_auc, fix_acc, m_acc, fix_f1, m_f1, fix_sen, m_sen, fix_spe, m_spe = valid(valid_loader, model, config)
    best_auc, best_m_f1, best_fix_f1 = best_index

    logging.info('Valid-Cls: Mean AUC: {:.4f}'.format(m_auc))
    logging.info('Valid-Cls: 0.5 ACC: {:.4f}'.format(m_acc))
    logging.info('Valid-Cls: 0.5 F1:  {:.4f}'.format(m_f1))
    logging.info('Valid-Cls: 0.5 SEN: {:.4f}'.format(m_sen))
    logging.info('Valid-Cls: 0.5 SPE: {:.4f}'.format(m_spe))
    logging.info('Valid-Cls: Dynamic ACC: {:.4f}'.format(fix_acc))
    logging.info('Valid-Cls: Dynamic F1:  {:.4f}'.format(fix_f1))
    logging.info('Valid-Cls: Dynamic SEN: {:.4f}'.format(fix_sen))
    logging.info('Valid-Cls: Dynamic SPE: {:.4f}'.format(fix_spe))
    # print_result('Valid-Cls: ACC for All Classes: ', all_acc, config['Data_CLASSES'])
    # print_result('Valid-Cls: AUC for All Classes: ', all_auc, config['Data_CLASSES'])
    # print_result('Valid-Cls: F1  for All Classes: ', all_f1,  config['Data_CLASSES'])

    # if m_acc >= best_acc:
    #     save_checkpoint(model, 'ValidFold' + str(config['valid_fold']) + '_' + config['arch'], epoch, _best='acc', best=m_acc)
    #     best_acc = m_acc
    if m_auc >= best_auc:
        save_checkpoint(model, 'ValidFold' + str(config['valid_fold']) + '_' + config['arch'], epoch, _best='auc',
                        best=m_auc)
        best_auc = m_auc
    if m_f1 >= best_m_f1:
        save_checkpoint(model, 'ValidFold' + str(config['valid_fold']) + '_' + config['arch'], epoch, _best='0.5_f1',
                        best=m_f1)
        best_m_f1 = m_f1
    if fix_f1 >= best_fix_f1:
        save_checkpoint(model, 'ValidFold' + str(config['valid_fold']) + '_' + config['arch'], epoch, _best='fix_f1',
                        best=fix_f1)
        best_fix_f1 = fix_f1
    # if m_spe >= best_spe:
    #     save_checkpoint(model, 'ValidFold' + str(config['valid_fold']) + '_' + config['arch'], epoch, _best='spe', best=m_spe)
    #     best_spe = m_spe

    return [best_auc, best_m_f1, best_fix_f1]


def train(train_loader, model, optimizer, epoch, config):
    losses = AverageMeter()
    clslosses = AverageMeter()
    cls_AUCs = AverageMeter()
    cls_F1s = AverageMeter()
    batch_time = AverageMeter()
    num_classes = len(config['Data_CLASSES'])

    model.train()
    end = time.time()
    epoch_iter = 0
    window_level = config['window_level']
    window_width = config['window_width']
    # window = config['window']
    # level = config['level']
    for i, (image, label, img_names) in enumerate(train_loader):

        with torch.autograd.set_detect_anomaly(True):
            image = image.cuda()
            # lung = lung.cuda()
            # ncov = ncov.cuda()
            label = label.cuda()
            label = label.unsqueeze(1)
            bs = image.size(0)

            cls_outs = model(image)
            cls_outs = torch.sigmoid(cls_outs)
            clsloss = cls_criterion(cls_outs, label)

            loss = clsloss

            losses.update(loss.item(), bs)
            clslosses.update(clsloss.item(), bs)

            optimizer.zero_grad()
            if config['Using_apex']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            epoch_iter += bs
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config['print_freq'] == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'clsLoss {clsloss.val:.4f} ({clsloss.avg:.4f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    clsloss=clslosses, loss=losses))


def valid(valid_loader, model, config):
    batch_time = AverageMeter()
    model.eval()

    num_classes = len(config['Data_CLASSES'])
    window_level = config['window_level']
    window_width = config['window_width']
    # window = config['window']
    # level = config['level']
    with torch.no_grad():
        end = time.time()
        for i, (image, label, img_names) in enumerate(valid_loader):
            bs = image.size(0)
            image = image.cuda()
            label = label.cuda()
            label = label.unsqueeze(1)

            cls_outs = model(image)
            probe = torch.sigmoid(cls_outs)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % (config['print_freq'] * config['batchsize']) == 0:
                logging.info('Valid-Cls: [{}/{}]\t'
                             'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                    i, len(valid_loader), batch_time=batch_time))

            if i == 0:
                y_true = label.cpu().detach().numpy()
                y_pred = probe.cpu().detach().numpy()
            else:
                y_true = np.concatenate((y_true, label.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, probe.cpu().detach().numpy()), axis=0)

        # y_true = np.expand_dims(y_true, axis=1)
        assert len(y_true.shape) == 2, "y_true shape error."
        m_auc, threshold = calculate_auc(y_pred, y_true, config, show_roc_curve=True)
        fix_acc, _ = accuracy(y_pred, y_true, config, threshold)
        fix_f1, _ = calculate_f1(y_pred, y_true, config, threshold)
        fix_sen, _ = calculate_sen(y_pred, y_true, config, threshold, show_confusion_matrix=True)
        fix_spe, _ = calculate_spe(y_pred, y_true, config, threshold)
        m_acc, _ = accuracy(y_pred, y_true, config, 0.5)
        m_f1, _ = calculate_f1(y_pred, y_true, config, 0.5)
        m_sen, _ = calculate_sen(y_pred, y_true, config, 0.5, show_confusion_matrix=False)
        m_spe, _ = calculate_spe(y_pred, y_true, config, 0.5)

        return m_auc, fix_acc, m_acc, fix_f1, m_f1, fix_sen, m_sen, fix_spe, m_spe


def percentile(tensor, q):
    k = 1 + round(.01 * float(q) * (tensor.numel() - 1))
    result = tensor.view(-1).kthvalue(k).values.item()
    return result

# def apply_window(image, window, level):
#     min_value = level - window // 2
#     max_value = level + window // 2
#     image[image < min_value] = min_value
#     image[image > max_value] = max_value
#     return image
#
# def normalization(image):
#     B, C, D, H, W = image.size()
#     image = image.view(B, -1)
#     img_min = image.min(dim=1, keepdim=True)[0]
#     img_max = image.max(dim=1, keepdim=True)[0]
#     norm = img_max - img_min
#     norm[norm == 0] = 1e-5
#     image = (image - img_min) / norm
#     image = image.view(B, C, D, H, W)
#     return image
