import numpy as np


def dice_coeff(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    smooth = 0.0001
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    if np.isnan(iou_score):
        iou_score = 1
    return iou_score


def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred))
    fn = np.sum(np.logical_and(y_true, 1 - y_pred))
    recall_val = tp / (tp + fn)
    if np.isnan(recall_val):
        recall_val = 1
    return recall_val


def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(1 - y_true, y_pred))
    precision_val = tp / (tp + fp)
    if np.isnan(precision_val):
        precision_val = 1
    return precision_val


def f1(precision_val, recall_val):
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    return f1_val


def angle_acc(act_angle, pred_angle):
    return (act_angle - pred_angle) ** 2


def dist_acc(pred_point1, pred_point2, act_point1, act_point2, size):

    center = np.array(size) / 2

    pred_point1 = np.array(pred_point1)
    pred_point2 = np.array(pred_point2)

    pred_dist = np.cross(pred_point2 - pred_point1, center - pred_point1) / np.linalg.norm(pred_point2 - pred_point1)

    act_point1 = np.array(act_point1)
    act_point2 = np.array(act_point2)

    act_dist = np.cross(act_point2 - act_point1, center - act_point1) / np.linalg.norm(act_point2 - act_point1)

    return np.square(act_dist - pred_dist)
