import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from pytorch examples
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: First box in [x1, y1, x2, y2] format
        box2: Second box in [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision (AP) from precision-recall curve
    
    Args:
        recalls: List of recall values
        precisions: List of precision values
        
    Returns:
        AP value
    """
    # Make sure lists are numpy arrays
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # Add sentinel values for easier calculation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope - highest precision for each recall level
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum area under curve (using rectangles)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_map(detections, annotations, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        detections: List of detection dictionaries with 'boxes', 'scores', 'labels'
        annotations: List of annotation dictionaries with 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        mAP value
    """
    # Group detections and annotations by class
    class_detections = defaultdict(list)
    class_annotations = defaultdict(list)
    
    # Process all frames
    for frame_dets, frame_annos in zip(detections, annotations):
        # Process detections
        for box, score, label in zip(frame_dets['boxes'], frame_dets['scores'], frame_dets['labels']):
            class_detections[label].append({
                'box': box,
                'score': score,
                'matched': False  # Flag for matching to ground truth
            })
        
        # Process annotations
        for box, label in zip(frame_annos['boxes'], frame_annos['labels']):
            class_annotations[label].append({
                'box': box,
                'matched': False  # Flag for matching to detection
            })
    
    # Calculate AP for each class
    aps = []
    
    for class_id in class_detections.keys():
        # Get detections and annotations for this class
        dets = class_detections[class_id]
        annos = class_annotations.get(class_id, [])
        
        # Sort detections by score (descending)
        dets.sort(key=lambda x: x['score'], reverse=True)
        
        # If no annotations for this class, AP = 0
        if not annos:
            if dets:  # if we have detections but no ground truth, AP = 0
                aps.append(0.0)
            continue
        
        # Initialize precision/recall lists
        precisions = []
        recalls = []
        
        # Count true positives and false positives
        true_positives = 0
        false_positives = 0
        
        # Reset matched flags
        for anno in annos:
            anno['matched'] = False
        
        # Process each detection
        for det in dets:
            # Find best matching ground truth
            best_iou = -1
            best_anno_idx = -1
            
            for i, anno in enumerate(annos):
                if anno['matched']:
                    continue
                
                iou = calculate_iou(det['box'], anno['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_anno_idx = i
            
            # Check if match is valid
            if best_iou >= iou_threshold:
                # Mark as matched
                annos[best_anno_idx]['matched'] = True
                true_positives += 1
            else:
                false_positives += 1
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / len(annos)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP for this class
        if precisions and recalls:
            ap = calculate_ap(recalls, precisions)
            aps.append(ap)
    
    # Calculate mAP
    if aps:
        return sum(aps) / len(aps)
    else:
        return 0.0

def calculate_map_at_iou(detections, annotations, iou_thresholds=[0.5, 0.75]):
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds
    
    Args:
        detections: List of detection dictionaries with 'boxes', 'scores', 'labels'
        annotations: List of annotation dictionaries with 'boxes', 'labels'
        iou_thresholds: List of IoU thresholds for evaluation
        
    Returns:
        Dict mapping threshold to mAP value
    """
    result = {}
    
    for threshold in iou_thresholds:
        mAP = calculate_map(detections, annotations, iou_threshold=threshold)
        result[f'mAP@{threshold}'] = mAP
    
    # Calculate average mAP across all thresholds
    result['mAP'] = sum(result.values()) / len(result)
    
    return result
