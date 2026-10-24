import contextlib
import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# OWL-ViT / CLIP normalisation constants (Owlv2ImageProcessor defaults)
OWL_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWL_STD = (0.26862954, 0.26130258, 0.27577711)


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


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def owl_preprocess(image, image_size: int, mean=OWL_MEAN, std=OWL_STD) -> torch.Tensor:
    """
    OWLv2 preprocessing: pad the image to a square (bottom/right, grey 0.5),
    resize to image_size x image_size and normalise with CLIP statistics.
    Boxes predicted by OWLv2 are therefore normalised w.r.t. the padded square,
    i.e. pixel_coord = normalised_coord * max(H, W).

    Args:
        image: PIL image (RGB)
        image_size: Target side length (960 for OWLv2-B/16, 1008 for OWLv2-L/14)

    Returns:
        pixel_values: Tensor [3, image_size, image_size]
    """
    x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    _, h, w = x.shape
    side = max(h, w)
    x = F.pad(x, (0, side - w, 0, side - h), value=0.5)
    x = F.interpolate(x[None], size=(image_size, image_size), mode='bilinear',
                      align_corners=False, antialias=True)[0]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (x - mean) / std


def owl_unnormalize(pixel_values: torch.Tensor, mean=OWL_MEAN, std=OWL_STD) -> torch.Tensor:
    """Inverse of the normalisation in owl_preprocess (for visualisation)"""
    mean = torch.tensor(mean, device=pixel_values.device).view(3, 1, 1)
    std = torch.tensor(std, device=pixel_values.device).view(3, 1, 1)
    return (pixel_values * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU between two sets of xyxy boxes -> ([N, M] iou, [N, M] union)"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise generalised IoU (Rezatofighi et al. 2019) between xyxy boxes -> [N, M]"""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-7)


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: First box in [x1, y1, x2, y2] format
        box2: Second box in [x1, y1, x2, y2] format

    Returns:
        IoU value
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / (box1_area + box2_area - intersection_area)


# ---------------------------------------------------------------------------
# Evaluation (COCO-style AP, AP50, AP75 as in FSVOD (Fan et al. 2022))
# ---------------------------------------------------------------------------

class DetectionAccumulator:
    """
    Collects per-frame detections and ground truth and evaluates them with the
    COCO protocol (pycocotools). Every (episode, frame) pair is an image, and
    categories are the dataset's real category ids, so a class only has ground
    truth in the frames of episodes in which it was one of the N support classes.
    """
    def __init__(self):
        self.images = []
        self.annotations = []
        self.detections = []
        self.categories = {}

    def add_frame(self, image_size: Tuple[int, int], gt_boxes, gt_labels, det_boxes, det_scores, det_labels):
        """
        Args:
            image_size: (H, W) of the original frame
            gt_boxes: [M, 4] xyxy pixel boxes
            gt_labels: M category ids
            det_boxes: [D, 4] xyxy pixel boxes
            det_scores: D scores
            det_labels: D category ids
        """
        image_id = len(self.images) + 1
        h, w = image_size
        self.images.append({'id': image_id, 'height': int(h), 'width': int(w)})
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            self.categories.setdefault(label, len(self.categories) + 1)
            self.annotations.append({
                'id': len(self.annotations) + 1, 'image_id': image_id,
                'category_id': self.categories[label], 'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0
            })
        for box, score, label in zip(det_boxes, det_scores, det_labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            self.categories.setdefault(label, len(self.categories) + 1)
            self.detections.append({
                'image_id': image_id, 'category_id': self.categories[label],
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)
            })

    def summarize(self, per_class: bool = False, verbose: bool = False) -> Dict:
        """Returns {'AP', 'AP50', 'AP75'} (in %), optionally with per-class AP"""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self.annotations:
            return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'num_images': len(self.images), 'num_gt': 0}

        gt = COCO()
        gt.dataset = {
            'images': self.images,
            'annotations': self.annotations,
            'categories': [{'id': cid, 'name': str(name)} for name, cid in self.categories.items()]
        }
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            dt = gt.loadRes(self.detections) if self.detections else COCO()
            if not self.detections:
                dt.dataset = {'images': self.images, 'annotations': [], 'categories': gt.dataset['categories']}
                dt.createIndex()
            coco_eval = COCOeval(gt, dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            coco_eval.summarize()
        if verbose:
            print(buf.getvalue())

        stats = coco_eval.stats
        results = {
            'AP': 100.0 * max(stats[0], 0.0),
            'AP50': 100.0 * max(stats[1], 0.0),
            'AP75': 100.0 * max(stats[2], 0.0),
            'num_images': len(self.images),
            'num_gt': len(self.annotations),
            'num_det': len(self.detections)
        }

        if per_class:
            # precision: [T(iou), R, K, A, M]; use area='all', maxDets=100
            precision = coco_eval.eval['precision']
            id_to_name = {cid: name for name, cid in self.categories.items()}
            per_class_ap = {}
            for k_idx, cat_id in enumerate(coco_eval.params.catIds):
                p = precision[:, :, k_idx, 0, -1]
                p = p[p > -1]
                if p.size:
                    per_class_ap[str(id_to_name[cat_id])] = 100.0 * float(p.mean())
            results['per_class'] = per_class_ap

        return results
