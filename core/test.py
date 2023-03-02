import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metrics.metrics import recall, precision, f1, dice_coeff, iou
from utils.img_utils import max_contour


def test(model, dataloaders):
    was_training = model.training
    model.eval()

    dice_arr = []
    iou_arr = []
    precision_arr = []
    recall_arr = []
    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                img = inputs.cpu().data[j].squeeze()
                label = labels.cpu().data[j].squeeze()
                label = label.squeeze().cpu().numpy()
                pred_ = outputs.cpu().data[j]
                pred = pred_.squeeze()
                out2 = pred_.data.max(0)[1].squeeze_(1)
                pred = F.softmax(pred, dim=0)
                pred = pred.squeeze().cpu().numpy()

                dice_arr_per_label_type = []
                iou_arr_per_label_type = []
                precision_arr_per_label_type = []
                recall_arr_per_label_type = []
                for label_type in range(1, pred.shape[0], 1):
                    # the below commented lines are for needle dataset

                    # pred_point1, pred_point2, pred_angle = max_contour(pred)
                    # act_point1, act_point2, act_angle = max_contour(label)

                    # pred_rect = inline_BB(pred_point1, pred_point2, pred.shape)/255
                    # act_rect = inline_BB(act_point1, act_point2, pred.shape)/255

                    # post_processing = pred_needle_img(pred_point1, pred_point2, size = pred.shape)

                    # dist_acc_arr.append(dist_acc(pred_point1, pred_point2, act_point1, act_point2, pred.shape))
                    # angle_acc_arr.append(angle_acc(pred_angle, act_angle))

                    # dice_arr.append(dice_coef(label, post_processing / 255))

                    pred_per_label_type = pred[label_type]
                    # pred_per_label_type[pred_per_label_type > 0.5] = 1
                    # pred_per_label_type[pred_per_label_type <= 0.5] = 0
                    # post_process = max_contour(pred_per_label_type)
                    post_process = pred_per_label_type
                    # dice_arr_per_label_type.append(dice_coeff(label, post_process / 255))
                    # iou_arr_per_label_type.append(iou(label, post_process / 255))
                    # print(iou(label, out2==label_type))
                    dice_arr_per_label_type.append(dice_coeff(label, out2 == label_type))
                    iou_arr_per_label_type.append(iou(label, out2 == label_type))
                    precision_arr_per_label_type.append(precision(label, out2 == label_type))
                    recall_arr_per_label_type.append(recall(label, out2 == label_type))

                dice_arr.append(np.mean(dice_arr_per_label_type))
                iou_arr.append(np.mean(iou_arr_per_label_type))
                precision_arr.append(np.mean(precision_arr_per_label_type))
                recall_arr.append(np.mean(recall_arr_per_label_type))

        # print("-" * 10 + "Avg Dice" + "-" * 10)
        # print(np.mean(dice_arr))
        #
        # print("-" * 10 + "Avg IOU" + "-" * 10)
        # print(np.mean(iou_arr))
        #
        # print("-" * 10 + "Avg Precision" + "-" * 10)
        # print(np.mean(precision_arr))
        #
        # print("-" * 10 + "Avg Recall" + "-" * 10)
        # print(np.mean(recall_arr))
        #
        # print("-" * 10 + "F1 Score" + "-" * 10)
        # print(f1(np.mean(precision_arr), np.mean(recall_arr)))

        model.train(mode=was_training)

    return {
        "dice": np.mean(dice_arr),
        "iou": np.mean(iou_arr),
        "precision": np.mean(precision_arr),
        "recall": np.mean(recall_arr),
        "f1": f1(np.mean(precision_arr), np.mean(recall_arr)),
    }


def visualize_results(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():

        figure, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(15, 3.75 * num_images))

        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                img = inputs.cpu().data[j].squeeze()
                label = labels.cpu().data[j].squeeze()
                label = label.squeeze().cpu().numpy()
                pred = outputs.cpu().data[j].squeeze()
                pred = nn.Softmax()(pred)[1]
                pred = pred.squeeze().cpu().numpy()

                axes[j, 0].imshow(np.transpose(img, (1, 2, 0)), cmap="gray")
                axes[j, 1].imshow(pred, cmap="gray")
                pred[pred > 0.5] = 255
                pred[pred <= 0.5] = 0
                # post_process = max_contour(pred)
                axes[j, 2].imshow(pred, cmap="gray")
                axes[j, 3].imshow(label, cmap="gray")
                cols = ["Input", "Prediction", "Post-Process", "Ground Truth"]

                for ax, col in zip(axes[0], cols):
                    ax.set_title(col)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    figure.tight_layout()
                    return
        model.train(mode=was_training)
