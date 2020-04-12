import os
import shutil
import torch
import logging
import itertools
from collections import OrderedDict
from sklearn import metrics
import math
import numpy as np
import torch.optim
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# Some default settings
model_savepath = 'checkpoints'
outputs_path = 'outputs'
power = 0.9
cls_thresh = 0.5


def plot_roc_curve(fpr, tpr, auc, save_path):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outputs_path, 'roc.png'), dpi=100)
    plt.close()


def plot_confusion_matrix(cm, classes, save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=100)
    plt.close()


def get_one_hot(data, num_classes):
    data = data.reshape(-1).astype(np.int64)
    data = np.eye(num_classes)[data]
    return data


def calculate_auc(y_pred, y_gt, config, show_roc_curve=False):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the mean AUC'''
    preds = y_pred
    label = y_gt

    fpr, tpr, thresholds = metrics.roc_curve(label, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    diff = abs(tpr - fpr)
    max_diff_index = np.argwhere(diff == max(diff))[0]
    # global cls_thresh
    best_thresholds = thresholds[max_diff_index][0]
    logging.info("threshold: {}".format(best_thresholds))

    if show_roc_curve:
        plot_roc_curve(fpr, tpr, auc, os.path.join(outputs_path, 'roc_valid{}.png'.format(config['valid_fold'])))

    mean_auc = auc

    return mean_auc, best_thresholds


def calculate_f1(output, target, config, threshold):
    num_classes = len(config['Data_CLASSES'])
    '''calculate the f1 score'''
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred[y_pred >= threshold] = 1.0
    y_pred[y_pred <= threshold] = 0.0
    y_true_one_hot = get_one_hot(y_true, num_classes)
    y_pred_one_hot = get_one_hot(y_pred, num_classes)

    f1_each_class = []
    nan_index = []
    mean_f1 = 0.0

    for index in range(num_classes):

        preds = y_pred_one_hot[:, index]
        label = y_true_one_hot[:, index]

        f1 = metrics.f1_score(label, preds, average='binary')

        if math.isnan(f1):
            nan_index.append(index)
            f1 = 0.0

        f1_each_class.append(f1)
        mean_f1 += f1

    # mean_f1 = metrics.f1_score(y_gt.flatten(), y_pred.flatten(), average='binary')
    mean_f1 = mean_f1 / num_classes

    return mean_f1, f1_each_class


def accuracy(output, target, config, threshold):
    num_classes = len(config['Data_CLASSES'])
    """Computes the precision@k for the specified values of k"""
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred[y_pred >= threshold] = 1.0
    y_pred[y_pred <= threshold] = 0.0
    y_pred_one_hot = get_one_hot(y_pred, num_classes)
    y_true_one_hot = get_one_hot(y_true, num_classes)

    error_map = np.equal(y_pred_one_hot, y_true_one_hot)
    acc_each_class = []
    nan_index = []
    mean_acc = 0.0
    bs = float(output.shape[0])

    for class_index in range(num_classes):
        class_error_map = error_map[:, class_index]
        acc = np.sum(class_error_map) / bs
        if math.isnan(acc):
            nan_index.append(class_index)
            acc = 0.0
        acc_each_class.append(acc)
        mean_acc += acc

    mean_acc = mean_acc / num_classes

    return mean_acc, acc_each_class


def calculate_sen(output, target, config, threshold, show_confusion_matrix=False):
    """Computes the precision@k for the specified values of k"""
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred[y_pred >= threshold] = 1.0
    y_pred[y_pred <= threshold] = 0.0
    classes = config['Data_CLASSES']
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    if show_confusion_matrix:
        # calc confusion matrix
        save_path = os.path.join(outputs_path, 'cm_valid{}_thres{}.png'.format(config['valid_fold'], threshold))
        plot_confusion_matrix(cm, classes, save_path)
        # f, ax = plt.subplots()
        # sns.heatmap(cm, annot=True, ax=ax)
        # ax.set_title('Confusion Matrix')
        # ax.set_xlabel('Predict')
        # ax.set_ylabel('True')
        # f.savefig(save_path, bbox_inches='tight')
        # f.close()

    recall = cm[1, 1] / np.sum(cm[1, :])
    mean_recall = recall

    return mean_recall, 0


def calculate_spe(output, target, config, threshold):
    """Computes the precision@k for the specified values of k"""
    y_pred = deepcopy(output)
    y_true = deepcopy(target)
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred[y_pred >= threshold] = 1.0
    y_pred[y_pred <= threshold] = 0.0
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    precision = cm[0, 0] / np.sum(cm[0, :])
    mean_precision = precision

    return mean_precision, 0


def print_result(display_str, result_class, classes):
    num_classes = len(classes)
    display_str = display_str
    for i in range(num_classes):
        if i < num_classes - 1:
            display_str += '{} {:.3f}, '.format(classes[i], result_class[i])
        else:
            display_str += '{} {:.3f}'.format(classes[i], result_class[i])
    logging.info(display_str)


def save_checkpoint(net, arch, epoch, _best=None, best=0):
    savepath = os.path.join(model_savepath, arch)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_name = os.path.join(savepath, "{}_epoch_{:0>4}".format(arch, epoch) + '.pth')
    torch.save({
        'model': net.state_dict(),
        'epoch': epoch,
    }, file_name)
    remove_flag = False
    if _best:
        best_name = os.path.join(savepath, "{}_best_{}".format(arch, _best) + '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_{}".format(arch, _best) + '.txt'), 'w')
        file.write('arch: {}'.format(arch) + '\n')
        file.write('epoch: {}'.format(epoch) + '\n')
        file.write('best {}: {}'.format(_best, best) + '\n')
        file.close()
    if remove_flag:
        os.remove(file_name)


def load_checkpoint(net, model_path, _sgpu=True):
    state_dict = torch.load(model_path)
    if _sgpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            # print(k)
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            head = k[:7]
            if head != 'module.':
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    logging.info('Finish loading resume network')


def set_requires_grad(net, fixed_layer, _sgpu=True):
    update_flag = {}
    for name, _ in net.named_parameters():
        # print(name)
        update_flag[name] = 0
        for item in fixed_layer:
            if _sgpu:
                if name[:len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1
            else:
                if name[7:7 + len(item)] == item:
                    # print('hehe')
                    update_flag[name] = 1

    for name, param in net.named_parameters():
        # print(name)
        if update_flag[name] == 1:
            param.requires_grad = False
        else:
            param.requires_grad = True


def adjust_learning_rate(optimizer, epoch, epoch_num, initial_lr, reduce_epoch, decay=0.1):
    if reduce_epoch == 'dynamic':
        lr = initial_lr * (1 - math.pow(float(epoch) / float(epoch_num), power))
    else:
        lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count
