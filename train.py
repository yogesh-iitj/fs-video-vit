import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import random

from model import FewShotVideoObjectDetection
from dataloader import get_fsvod_loaders
from utils import AverageMeter, calculate_map, calculate_iou

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, data_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    
    # Meters for tracking statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    box_losses = AverageMeter()
    
    end = time.time()
    
    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Get data
        support_images = {cls_id: imgs.to(device) for cls_id, imgs in batch['support_images'].items()}
        query_frames = batch['query_frames'].to(device)
        annotations = batch['frame_annotations']
        classes = batch['classes']
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            frames=query_frames,
            support_images=support_images,
            return_all_frames=True
        )
        
        # Calculate loss for each frame
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        
        for t, (frame_output, frame_anno) in enumerate(zip(outputs, annotations)):
            # Get ground truth boxes and classes
            gt_boxes = []
            gt_classes = []
            
            for obj in frame_anno['objects']:
                gt_boxes.append(obj['bbox'])
                gt_classes.append(obj['category_id'])
            
            # If no objects in this frame, skip loss calculation
            if not gt_boxes:
                continue
            
            # Compute classification loss and box regression loss
            cls_loss, box_loss = criterion(frame_output, gt_boxes, gt_classes)
            
            frame_loss = cls_loss + box_loss
            total_loss += frame_loss
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
        
        # Average loss over frames
        num_frames = len(outputs)
        if num_frames > 0:
            total_loss /= num_frames
            total_cls_loss /= num_frames
            total_box_loss /= num_frames
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        # Update statistics
        losses.update(total_loss.item(), query_frames.size(0))
        cls_losses.update(total_cls_loss, query_frames.size(0))
        box_losses.update(total_box_loss, query_frames.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print statistics periodically
        if i % 10 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Cls {cls_losses.val:.4f} ({cls_losses.avg:.4f}) '
                  f'Box {box_losses.val:.4f} ({box_losses.avg:.4f})')
    
    return {
        'loss': losses.avg,
        'cls_loss': cls_losses.avg,
        'box_loss': box_losses.avg
    }

def validate(model, data_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    # Meters for tracking statistics
    batch_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    box_losses = AverageMeter()
    
    # For mAP calculation
    all_detections = []
    all_annotations = []
    
    end = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Validating")):
            # Get data
            support_images = {cls_id: imgs.to(device) for cls_id, imgs in batch['support_images'].items()}
            query_frames = batch['query_frames'].to(device)
            annotations = batch['frame_annotations']
            classes = batch['classes']
            
            # Forward pass
            outputs = model(
                frames=query_frames,
                support_images=support_images,
                return_all_frames=True
            )
            
            # Calculate loss for each frame
            total_loss = 0
            total_cls_loss = 0
            total_box_loss = 0
            
            # Process each frame for evaluation
            for t, (frame_output, frame_anno) in enumerate(zip(outputs, annotations)):
                # Get ground truth boxes and classes
                gt_boxes = []
                gt_classes = []
                
                for obj in frame_anno['objects']:
                    gt_boxes.append(obj['bbox'])
                    gt_classes.append(obj['category_id'])
                
                # If no objects in this frame, skip loss calculation
                if not gt_boxes:
                    continue
                
                # Compute classification loss and box regression loss
                cls_loss, box_loss = criterion(frame_output, gt_boxes, gt_classes)
                
                frame_loss = cls_loss + box_loss
                total_loss += frame_loss
                total_cls_loss += cls_loss.item()
                total_box_loss += box_loss.item()
                
                # Process detections for mAP calculation
                # Get image size
                image_size = query_frames[t].shape[-2:]  # H, W
                
                # Get post-processed detections
                detections = model.postprocess_detections(
                    frame_output,
                    image_size,
                    confidence_threshold=0.5
                )
                
                # Append to lists for mAP calculation
                all_detections.append(detections)
                all_annotations.append({
                    'boxes': gt_boxes,
                    'labels': gt_classes
                })
            
            # Average loss over frames
            num_frames = len(outputs)
            if num_frames > 0:
                total_loss /= num_frames
                total_cls_loss /= num_frames
                total_box_loss /= num_frames
            
            # Update statistics
            losses.update(total_loss.item(), query_frames.size(0))
            cls_losses.update(total_cls_loss, query_frames.size(0))
            box_losses.update(total_box_loss, query_frames.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    # Calculate mAP
    mAP = calculate_map(all_detections, all_annotations)
    
    print(f' * Val Loss {losses.avg:.4f}\tCls {cls_losses.avg:.4f}\tBox {box_losses.avg:.4f}\tmAP {mAP:.4f}')
    
    return {
        'loss': losses.avg,
        'cls_loss': cls_losses.avg,
        'box_loss': box_losses.avg,
        'mAP': mAP
    }

class LossFunction(nn.Module):
    """Combined loss function for object detection"""
    def __init__(self, lambda_cls=1.0, lambda_box=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.box_loss_fn = nn.SmoothL1Loss()
    
    def forward(self, outputs, gt_boxes, gt_classes):
        """
        Calculate loss
        
        Args:
            outputs: Model outputs
            gt_boxes: Ground truth boxes
            gt_classes: Ground truth classes
            
        Returns:
            cls_loss, box_loss
        """
        # Extract classification scores and boxes
        cls_scores = outputs['cls_scores']
        pred_boxes = outputs['boxes']
        
        # Get device
        device = pred_boxes.device
        
        # Combine class scores for all classes
        all_cls_scores = torch.stack(list(cls_scores.values()), dim=-1)
        
        # Match predictions to ground truth using IoU
        num_preds = all_cls_scores.size(0)
        num_gts = len(gt_boxes)
        
        # If no ground truth or predictions, return zero loss
        if num_gts == 0 or num_preds == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Convert gt_boxes to tensor
        gt_boxes_tensor = torch.tensor(gt_boxes, device=device)
        
        # Calculate IoU between predictions and gt
        ious = torch.zeros((num_preds, num_gts), device=device)
        for i in range(num_preds):
            for j in range(num_gts):
                ious[i, j] = calculate_iou(pred_boxes[i].cpu().numpy(), gt_boxes[j])
        
        # Match predictions to ground truth
        matched_indices = []
        matched_gt_classes = []
        matched_gt_boxes = []
        
        # Greedy matching
        for _ in range(min(num_preds, num_gts)):
            # Find max IoU pair
            if ious.numel() == 0 or ious.max() < 0.5:
                break
                
            max_idx = ious.argmax()
            pred_idx = max_idx // num_gts
            gt_idx = max_idx % num_gts
            
            # Add to matched pairs
            matched_indices.append(pred_idx)
            matched_gt_classes.append(gt_classes[gt_idx])
            matched_gt_boxes.append(gt_boxes_tensor[gt_idx])
            
            # Mark this pair as matched by setting IoU to -1
            ious[pred_idx, :] = -1
            ious[:, gt_idx] = -1
        
        # Calculate classification loss
        cls_loss = torch.tensor(0.0, device=device)
        if matched_indices:
            # Convert matched classes to indices in cls_scores
            class_id_to_idx = {cls_id: i for i, cls_id in enumerate(cls_scores.keys())}
            matched_cls_indices = [class_id_to_idx[cls] for cls in matched_gt_classes]
            
            # Get classification scores for matched predictions
            matched_scores = all_cls_scores[matched_indices]
            
            # Calculate cross entropy loss
            target = torch.tensor(matched_cls_indices, device=device)
            cls_loss = self.cls_loss_fn(matched_scores, target)
        
        # Calculate box regression loss
        box_loss = torch.tensor(0.0, device=device)
        if matched_indices and matched_gt_boxes:
            # Get predicted boxes for matched predictions
            matched_pred_boxes = pred_boxes[matched_indices]
            
            # Convert to tensor
            matched_gt_boxes = torch.stack(matched_gt_boxes)
            
            # Calculate box loss
            box_loss = self.box_loss_fn(matched_pred_boxes, matched_gt_boxes)
        
        # Apply loss weights
        cls_loss = self.lambda_cls * cls_loss
        box_loss = self.lambda_box * box_loss
        
        return cls_loss, box_loss

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    data_loaders = get_fsvod_loaders(
        root_dir=args.data_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.img_size, args.img_size)
    )
    
    # Create model
    model = FewShotVideoObjectDetection(
        pretrained_model_name=args.pretrained_model,
        confidence_threshold=args.conf_threshold,
        temporal_confidence_threshold=args.temporal_threshold,
        freeze_encoder=not args.unfreeze_encoder
    )
    model.to(device)
    
    # Create loss function
    criterion = LossFunction(
        lambda_cls=args.lambda_cls,
        lambda_box=args.lambda_box
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_map = 0.0
    train_stats = []
    val_stats = []
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            data_loader=data_loaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        train_stats.append(train_metrics)
        
        # Validate
        val_metrics = validate(
            model=model,
            data_loader=data_loaders['val'],
            criterion=criterion,
            device=device
        )
        val_stats.append(val_metrics)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'args': vars(args)
        }
        
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Save best model
        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved best model with mAP: {best_map:.4f}")
    
    # Save final statistics
    stats = {
        'train': train_stats,
        'val': val_stats
    }
    
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Training completed. Best mAP: {best_map:.4f}")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model'])
    
    test_metrics = validate(
        model=model,
        data_loader=data_loaders['test'],
        criterion=criterion,
        device=device
    )
    
    # Save test results
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"Test mAP: {test_metrics['mAP']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Video Object Detection Training")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--n_way', type=int, default=5, help="Number of classes per episode")
    parser.add_argument('--k_shot', type=int, default=5, help="Number of examples per class")
    parser.add_argument('--max_frames', type=int, default=16, help="Maximum number of frames per video")
    
    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default="google/owlv2-base-patch16-ensemble",
                       help="Pretrained model name or path")
    parser.add_argument('--conf_threshold', type=float, default=0.5, 
                       help="Confidence threshold for detections")
    parser.add_argument('--temporal_threshold', type=float, default=0.85,
                       help="Threshold for temporal propagation")
    parser.add_argument('--unfreeze_encoder', action='store_true',
                       help="Unfreeze encoder parameters")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--lambda_cls', type=float, default=1.0, help="Classification loss weight")
    parser.add_argument('--lambda_box', type=float, default=1.0, help="Box loss weight")
    
    # Other parameters
    parser.add_argument('--img_size', type=int, default=640, help="Image size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args)
