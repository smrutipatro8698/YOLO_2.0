from __future__ import division

import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def initialise_weights(m):
	classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def rescale_bounding_box(box, dim, shape):
	h, w = shape
	x_pad = max(h-w, 0) * (dim/max(shape))
	y_pad = max(w-h, 0) * (dim/max(shape))
	h_unpad = dim - y_pad
	w_unpad = dim - x_pad

	box[:, 0] = ((box[:, 0] - x_pad // 2) / w_unpad) * w
    box[:, 1] = ((box[:, 1] - y_pad // 2) / h_unpad) * h
    box[:, 2] = ((box[:, 2] - x_pad // 2) / w_unpad) * w
    box[:, 3] = ((box[:, 3] - y_pad // 2) / h_unpad) * h

    return box

def convert(x):
	converted = x.new(x.shape)
	converted[..., 0] = x[..., 0] - x[..., 2] / 2
    converted[..., 1] = x[..., 1] - x[..., 3] / 2
    converted[..., 2] = x[..., 0] + x[..., 2] / 2
    converted[..., 3] = x[..., 1] + x[..., 3] / 2
    return converted

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics