import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List

from utils import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    One-to-one matching between the P patch predictions of a frame and its ground-truth
    objects (DETR, Carion et al. 2020): sigma_t(m) in the paper's training objective.
    """
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, probs: torch.Tensor, boxes: torch.Tensor, target: Dict[str, torch.Tensor]):
        """
        Args:
            probs: [P, N + 1] class probabilities of one frame
            boxes: [P, 4] predicted cxcywh boxes
            target: dict with boxes [M, 4] (cxcywh) and labels [M]

        Returns:
            (pred_indices, target_indices) LongTensors of length min(P, M)
        """
        tgt_boxes, tgt_labels = target['boxes'], target['labels']
        cost_class = -probs[:, tgt_labels]
        cost_bbox = torch.cdist(boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes))
        cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        pred_idx, tgt_idx = linear_sum_assignment(cost.cpu().numpy())
        return (torch.as_tensor(pred_idx, dtype=torch.long, device=probs.device),
                torch.as_tensor(tgt_idx, dtype=torch.long, device=probs.device))


class SetCriterion(nn.Module):
    """
    Training objective of the paper, summed over all frames of the episode:

        L = sum_t [ sum_m lambda_cls * L_cls(c^_t^{sigma_t(m)}, c_t^m) + sum_m lambda_box * L_box(b^_t^{sigma_t(m)}, b_t^m) ]

    L_cls is the cross-entropy over the N classes + background for every patch (unmatched
    patches are background). Because a frame has thousands of patches and only a few
    objects, the cross-entropy is focal-modulated (gamma > 0) and normalised by the number
    of objects; gamma = 0 recovers plain (DETR-style weighted) cross-entropy.
    L_box = L1 + generalised IoU on matched patches.
    """
    def __init__(self, lambda_cls: float = 2.0, lambda_box: float = 5.0, lambda_giou: float = 2.0,
                 focal_gamma: float = 2.0, background_weight: float = 1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_giou = lambda_giou
        self.focal_gamma = focal_gamma
        self.background_weight = background_weight
        self.matcher = HungarianMatcher(lambda_cls, lambda_box, lambda_giou)

    def forward(self, outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        """
        Args:
            outputs: Per-frame model outputs (batch size 1)
            targets: Per-frame targets with boxes [M, 4] (normalised cxcywh) and labels [M]

        Returns:
            dict with loss (total), loss_cls, loss_box, loss_giou
        """
        device = outputs[0]['logits'].device
        num_objects = sum(len(t['labels']) for t in targets)
        normaliser = max(float(num_objects), 1.0)

        loss_cls = torch.zeros((), device=device)
        loss_box = torch.zeros((), device=device)
        loss_giou = torch.zeros((), device=device)
        weight_sum = 0.0

        for frame_output, target in zip(outputs, targets):
            logits = frame_output['logits'][0]                      # [P, N + 1]
            boxes = frame_output['boxes'][0]                        # [P, 4]
            num_classes = logits.size(-1) - 1
            target_classes = torch.full((logits.size(0),), num_classes, dtype=torch.long, device=device)

            if len(target['labels']):
                tgt = {'boxes': target['boxes'].to(device), 'labels': target['labels'].to(device)}
                pred_idx, tgt_idx = self.matcher(frame_output['probs'][0].detach(), boxes.detach(), tgt)
                target_classes[pred_idx] = tgt['labels'][tgt_idx]

                matched_boxes = boxes[pred_idx]
                gt_boxes = tgt['boxes'][tgt_idx]
                loss_box = loss_box + F.l1_loss(matched_boxes, gt_boxes, reduction='sum')
                giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(matched_boxes),
                                                      box_cxcywh_to_xyxy(gt_boxes)))
                loss_giou = loss_giou + (1 - giou).sum()

            log_probs = logits.log_softmax(dim=-1)
            log_p_t = log_probs.gather(1, target_classes[:, None]).squeeze(1)
            ce = -log_p_t
            if self.focal_gamma > 0:
                ce = ce * (1 - log_p_t.exp()).pow(self.focal_gamma)
            weights = torch.ones_like(ce)
            weights[target_classes == num_classes] = self.background_weight
            loss_cls = loss_cls + (ce * weights).sum()
            weight_sum += float(weights.sum())

        if self.focal_gamma > 0:
            loss_cls = loss_cls / normaliser
        else:
            loss_cls = loss_cls / max(weight_sum, 1.0)
        loss_box = loss_box / normaliser
        loss_giou = loss_giou / normaliser

        loss_cls = self.lambda_cls * loss_cls
        loss_box = self.lambda_box * loss_box
        loss_giou = self.lambda_giou * loss_giou
        return {
            'loss': loss_cls + loss_box + loss_giou,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_giou': loss_giou
        }
