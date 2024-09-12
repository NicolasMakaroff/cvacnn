#import torch.nn as nn
#import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import f1_score
import optax
import jax.numpy as jnp
import numpy as np

def dice_coef_metric(inputs, target):
    intersection = 2.0 * jnp.sum(target*inputs) + 1e-6
    union = jnp.sum(target) + jnp.sum(inputs) +1e-6
    return intersection/union

def dice_coef_loss(inputs, target):
    smooth = 1.0 
    inputs = inputs.flatten()
    target = target.flatten() >= 1. * 1.
    intersection = 2.0 * jnp.sum(inputs*target) + smooth
    union = jnp.sum(target) + jnp.sum(inputs) + smooth
    return 1 - (intersection/union)

def binary_cross_entropy(inputs, target):
    eps = 1e-9
    inputs = jnp.clip(inputs, eps, 1-eps)
    loss = -(target * jnp.log(inputs) + (1 - target) * jnp.log(1 - inputs))
    return jnp.mean(loss)

def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_score = binary_cross_entropy(inputs, target)
    return bce_score + dice_score

def compute_iou(mask_pred, mask, threshold=0.3):

    norm_mask = mask >= 1. * 1.
    out_cut = jnp.copy(mask_pred)
    out_cut = out_cut >= threshold * 1.0

    picloss = dice_coef_metric(out_cut, norm_mask)

    return picloss

def hausdorff(inputs, target):

    # Convert the sets of points to numpy arrays
    set_a = np.array(jnp.squeeze(inputs)) > .3 * 1.
    set_b = np.array(jnp.squeeze(target))

    # Compute the directed Hausdorff distance
    hausdorff_distance = max(directed_hausdorff(set_a, set_b)[0], directed_hausdorff(set_b, set_a)[0])

    return hausdorff_distance

def F1(inputs, target):
    a = np.array(jnp.ravel(inputs)) > .3 * 1.
    b = np.array(jnp.ravel(target))

    return f1_score(a, b, average='binary')

def precision_recall_FP_FN(inputs, target):
    precision = jnp.sum(target*inputs)
    recall = jnp.sum(target) + jnp.sum(inputs)

    inputs = (jnp.ravel(jnp.squeeze(inputs)) > .1)
    #inputs = torch.int(inputs > .3 * 1)
    target = (jnp.ravel(jnp.squeeze(target)) >= 1.0 )
    TP = jnp.sum(target & inputs)
    FP = jnp.sum(inputs & ~target)
    TN = jnp.sum(~target & ~inputs)
    FN = jnp.sum(target & ~inputs)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    #TPR = TP / (TP + FN)
    #TNR = TN / (TN + FP)
    return [precision, recall, FPR, FNR]

