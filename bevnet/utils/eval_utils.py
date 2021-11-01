import numpy as np
import torch

def _1hot(y, num_classes):
    y_1hot = torch.BoolTensor(y.shape[0],
                              num_classes).to(device=y.device)
    y_1hot.zero_()
    y_1hot.scatter_(1, y[:, None].to(torch.int64), 1)
    return y_1hot


def _idiv(a, b):
    ignore = b == 0
    b[ignore] = 1
    div = a / b
    div[ignore] = np.nan
    return div


class NumpyEvaluator(object):
    def __init__(self, num_classes, ignore_label=255):
        self.labels = np.arange(num_classes)
        self.ignore_label = ignore_label
        self.conf_mat = None

    def reset(self):
        self.conf_mat = None

    def append(self, pred, label):
        valid = label != self.ignore_label

        pred, label = pred[valid], label[valid]

        conf = confusion_matrix(label.cpu(), pred.cpu(), labels=self.labels)

        if self.conf_mat is None:
            self.conf_mat = conf
        else:
            self.conf_mat += conf

    def classwiseAcc(self):
        tp = np.diag(self.conf_mat)
        total = self.conf_mat.sum(axis=0)
        return _idiv(tp, total)

    def acc(self):
        return np.nanmean(self.classwiseAcc())

    def classwiseIoU(self):
        tp = np.diag(self.conf_mat)
        fn = self.conf_mat.sum(axis=1) - tp
        tp_fp = self.conf_mat.sum(axis=0)
        return _idiv(tp, tp_fp + fn)

    def meanIoU(self):
        return np.nanmean(self.classwiseIoU())


class Evaluator(object):
    def __init__(self, num_classes, ignore_label=255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.tp = None

    def reset(self):
        self.tp = None

    def append(self, pred, label, valid_mask=None):
        valid = label != self.ignore_label

        if valid_mask is not None:
            valid &= valid_mask

        pred, label = pred[valid], label[valid]

        pred_1hot = _1hot(pred, self.num_classes)
        label_1hot = _1hot(label, self.num_classes)

        tp = (pred_1hot & label_1hot).sum(0)
        pred_total = pred_1hot.sum(0)
        label_total = label_1hot.sum(0)

        if self.tp is None:
            self.tp = tp
            self.pred_total = pred_total
            self.label_total = label_total
        else:
            self.tp += tp
            self.pred_total += pred_total
            self.label_total += label_total

    def classwiseAcc(self):
        acc = (self.tp / self.pred_total).cpu().numpy()

        if self.ignore_label < len(acc):
            acc = np.delete(acc, self.ignore_label)

        return acc

    def acc(self):
        return np.nanmean(self.classwiseAcc())

    def classwiseIoU(self):
        iou = self.tp / (self.pred_total + self.label_total - self.tp)
        iou = iou.cpu().numpy()
 
        if self.ignore_label < len(iou):
            iou = np.delete(iou, self.ignore_label)

        return iou

    def meanIoU(self):
        return np.nanmean(self.classwiseIoU())


