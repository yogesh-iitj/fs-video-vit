import os
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

from model import FewShotVideoObjectDetection
from dataloader import get_fsvod_loaders
from utils import calculate_map, calculate_map_at_iou

def visualize_detections(frame, detections, class_colors, output_path=None, threshold=0.5):
    """
    Visualize detections on a frame
    
    Args:
        frame: PIL Image object
        detections: Dictionary with 'boxes', 'scores', 'labels'
        class_colors: Dictionary mapping class IDs to colors
        output_path: Path to save visualization (optional)
        threshold: Confidence threshold for visualization
    
    Returns:
        Visualization as PIL Image
    """
    # Create a copy of the frame
    img = frame.copy()
    draw = ImageDraw.Draw(img)
    
    # Get frame dimensions
    width, height = img.size
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each detection
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
        if score < threshold:
            continue
        
        # Get color for this class
        color = class_colors.get(label, (255, 0, 0))
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label_text = f"{label}: {score:.2f}"
        text_w, text_h = draw.textsize(label_text, font=font)
        draw.rectangle([x1, y1, x1 + text_w, y1 + text_h], fill=color)
        draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)
    
    # Save image if output path is provided
    if output_path:
        img.save(output_path)
    
    return img

def evaluate_model(args):
    """Evaluate model on test set"""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    data_loaders = get_fsvod_loaders(
        root_dir=args.data_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_frames=args.max_frames,
        batch_size=1,
        num_workers=args.num_workers,
        target_size=(args.img_size, args.img_size)
    )
    test_loader = data_loaders['test']
    
    # Load model
    model = FewShotVideoObjectDetection(
        pretrained_model_name=args.pretrained_model,
        confidence_threshold=args.conf_threshold,
        temporal_confidence_threshold=args.temporal_threshold
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # For mAP calculation
    all_detections = []
    all_annotations = []
    
    # For per-class mAP
    class_detections = {}
    class_annotations = {}
    
    # Random colors for visualization
    np.random.seed(42)
    class_colors = {}
    
    # Evaluate on test set
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get data
            support_images = {cls_id: imgs.to(device) for cls_id, imgs in batch['support_images'].items()}
            query_frames = batch['query_frames'].to(device)
            annotations = batch['frame_annotations']
            classes = batch['classes']
            
            # Create colors for classes
            for cls_id in classes:
                if cls_id not in class_colors:
                    class_colors[cls_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Forward pass
            outputs = model(
                frames=query_frames,
                support_images=support_images,
                return_all_frames=True
            )
            
            # Process each frame
            for t, (frame_output, frame_anno) in enumerate(zip(outputs, annotations)):
                # Get image size
                image_size = query_frames[t].shape[-2:]  # H, W
                
                # Get post-processed detections
                detections = model.postprocess_detections(
                    frame_output,
                    image_size,
                    confidence_threshold=args.conf_threshold
                )
                
                # Prepare ground truth annotations
                gt_boxes = []
                gt_classes = []
                
                for obj in frame_anno['objects']:
                    gt_boxes.append(obj['bbox'])
                    gt_classes.append(obj['category_id'])
                
                # Append to lists for mAP calculation
                all_detections.append(detections)
                all_annotations.append({
                    'boxes': gt_boxes,
                    'labels': gt_classes
                })
                
                # Update per-class detections and annotations
                for cls_id in classes:
                    # Filter detections for this class
                    cls_dets = {
                        'boxes': [box for box, lbl in zip(detections['boxes'], detections['labels']) if lbl == cls_id],
                        'scores': [score for score, lbl in zip(detections['scores'], detections['labels']) if lbl == cls_id],
                        'labels': [lbl for lbl in detections['labels'] if lbl == cls_id]
                    }
                    
                    # Filter annotations for this class
                    cls_annos = {
                        'boxes': [box for box, lbl in zip(gt_boxes, gt_classes) if lbl == cls_id],
                        'labels': [lbl for lbl in gt_classes if lbl == cls_id]
                    }
                    
                    # Initialize lists if needed
                    if cls_id not in class_detections:
                        class_detections[cls_id] = []
                        class_annotations[cls_id] = []
                    
                    # Append to lists
                    class_detections[cls_id].append(cls_dets)
                    class_annotations[cls_id].append(cls_annos)
                
                # Visualize if requested
                if args.visualize and i < args.num_vis_episodes:
                    # Convert tensor to PIL image
                    frame_tensor = query_frames[t].cpu()
                    frame_tensor = frame_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    frame_tensor = frame_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    frame_tensor = frame_tensor.clamp(0, 1)
                    
                    frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                    
                    # Visualize detections
                    vis_img = visualize_detections(
                        frame=frame_pil,
                        detections=detections,
                        class_colors=class_colors,
                        output_path=vis_dir / f"episode_{i}_frame_{t}.jpg",
                        threshold=args.conf_threshold
                    )
    
    # Calculate overall mAP at different IoU thresholds
    mAP_results = calculate_map_at_iou(
        all_detections, 
        all_annotations,
        iou_thresholds=[0.5, 0.75]
    )
    
    print("\nOverall mAP Results:")
    for k, v in mAP_results.items():
        print(f"{k}: {v:.4f}")
    
    # Calculate per-class mAP
    per_class_results = {}
    for cls_id in class_detections.keys():
        cls_map = calculate_map(class_detections[cls_id], class_annotations[cls_id])
        per_class_results[cls_id] = cls_map
    
    print("\nPer-class mAP Results:")
    for cls_id, mAP in per_class_results.items():
        print(f"Class {cls_id}: {mAP:.4f}")
    
    # Save results
    results = {
        'overall': mAP_results,
        'per_class': per_class_results
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")
    
    # Plot mAP results
    if args.plot_results:
        # Plot overall mAP
        plt.figure(figsize=(10, 6))
        plt.bar(mAP_results.keys(), mAP_results.values())
        plt.title('Overall mAP at different IoU thresholds')
        plt.ylabel('mAP')
        plt.savefig(output_dir / "overall_map.png")
        
        # Plot per-class mAP
        plt.figure(figsize=(12, 6))
        plt.bar(per_class_results.keys(), per_class_results.values())
        plt.title('Per-class mAP')
        plt.xlabel('Class ID')
        plt.ylabel('mAP')
        plt.savefig(output_dir / "per_class_map.png")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot Video Object Detection Model")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--n_way', type=int, default=5, help="Number of classes per episode")
    parser.add_argument('--k_shot', type=int, default=5, help="Number of examples per class")
    parser.add_argument('--max_frames', type=int, default=16, help="Maximum number of frames per video")
    
    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default="google/owlv2-large-patch16",
                       help="Pretrained model name or path")
    parser.add_argument('--conf_threshold', type=float, default=0.5, 
                       help="Confidence threshold for detections")
    parser.add_argument('--temporal_threshold', type=float, default=0.94,
                       help="Threshold for temporal propagation")
    
    # Evaluation parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument('--img_size', type=int, default=640, help="Image size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
    parser.add_argument('--output_dir', type=str, default="evaluation_results", help="Output directory")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help="Visualize detections")
    parser.add_argument('--num_vis_episodes', type=int, default=5,
                       help="Number of episodes to visualize")
    parser.add_argument('--plot_results', action='store_true', help="Plot mAP results")
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()
