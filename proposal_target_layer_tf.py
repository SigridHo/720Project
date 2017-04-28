import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import pdb

def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
    ''' Input: Proposal ROIs (0, x1, y1, x2, y2) coming from RPN, 
                Ground truth boxes (x1, y1, x2, y2, class),
                Number of classification classes
        Output: Selected ROIs, corresponding labels, bounding box targets,
                bbox_inside_weights,bbox_outside_weights'''
    # Make the format of gt_boxes same as ROIs
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    zeros_gt_boxes = np.concatenate((zeros, gt_boxes[:, :-1]), axis=1)
    
    # Comprise the ROIs and gt_boxes
    all_rois = np.concatenate((rpn_rois, zeros_gt_boxes), axis=0)

    # TRAIN.BATCH_SIZE is the minibatch size (number of regions of interest [ROIs])
    rois_per_image = cfg.TRAIN.BATCH_SIZE
    # FG_FRACTION is the fraction of minibatch that is labeled foreground
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Random sample the candidate ROIs. 
    # Get corresponding labels, bbox_targets and bbox_inside_weights
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)
    rois = rois.reshape(-1,5)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    ''' Generate samples of ROIs '''
    # overlaps: proportion of overlapping area and total area of two boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    # overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))

    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # N * (class, tx, ty, tw, th)
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)
    return labels, rois, bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    '''Compute bounding-box regression targets for an image.'''
    # targets: trasformation from ex_rois to gt_rois, (tx, ty, tw, th)
    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.concatenate(
            (labels[:, np.newaxis], targets), axis=1).astype(np.float32, copy=False)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    '''
    Returns:
        bbox_target (ndarray): N x 4C blob of regression targets
        bbox_inside_weights (ndarray): N x 4C blob of loss weights
    '''
    labels = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((labels.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(labels > 0)[0]
    for ind in inds:
        cls = labels[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
