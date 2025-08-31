import numpy as np
# import cv2
import torch

# Source: https://github.com/dmitrysarov/clDice/


def dice_loss(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate dice loss per batch and channel of sample.
    E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
    '''
    smooth = 1.
    iflat = pred.view(*pred.shape[:2], -1) #batch, channel, -1
    tflat = target.view(*target.shape[:2], -1)
    intersection = (iflat * tflat).sum(-1)
    return -((2. * intersection + smooth) /
              (iflat.sum(-1) + tflat.sum(-1) + smooth))

def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)

def soft_cldice_loss(pred, target, target_skeleton=None):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    cl_pred = soft_skeletonize(pred)
    if target_skeleton is None:
        target_skeleton = soft_skeletonize(target)
    iflat = norm_intersection(cl_pred, target)
    tflat = norm_intersection(target_skeleton, pred)
    intersection = iflat * tflat
    loss = -((2. * intersection) /
              (iflat + tflat))
    return loss.mean()