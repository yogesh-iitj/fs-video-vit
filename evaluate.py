import os
import time
import torch
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from model import load_model_from_checkpoint
from dataloader import get_fsvod_loader
from utils import DetectionAccumulator, owl_unnormalize


def autocast_context(device, enabled):
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def visualize_detections(frame, detections, class_names, class_colors, output_path=None, threshold=0.5):
    """
    Visualize detections on a frame

    Args:
        frame: PIL Image object
        detections: Dictionary with 'boxes' (xyxy pixels), 'scores', 'labels' (class index)
        class_names: List mapping class index to a name
        class_colors: List mapping class index to a color
        output_path: Path to save visualization (optional)
        threshold: Confidence threshold for visualization

    Returns:
        Visualization as PIL Image
    """
    img = frame.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for box, score, label in zip(detections['boxes'].tolist(), detections['scores'].tolist(),
                                 detections['labels'].tolist()):
        if score < threshold:
            continue
        color = class_colors[label]
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{class_names[label]}: {score:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)

    if output_path:
        img.save(output_path)
    return img


@torch.no_grad()
def run_evaluation(model, data_loader, device, amp=True, score_threshold=0.0, max_detections=100,
                   nms_iou=None, per_class=False, desc="Evaluating", vis_dir=None, num_vis_episodes=0,
                   vis_threshold=None, category_names=None):
    """
    Run the model over fixed evaluation episodes and compute COCO-style AP / AP50 / AP75.

    Returns:
        dict with AP, AP50, AP75 (%), counts, avg_memory_tokens, fps (and per_class)
    """
    model.eval()
    accumulator = DetectionAccumulator()
    memory_tokens = []
    num_frames = 0
    start = time.time()
    rng = np.random.RandomState(42)
    category_names = category_names or {}

    for i, batch in enumerate(tqdm(data_loader, desc=desc)):
        if not batch['targets']:
            continue
        support = batch['support_pixel_values'].to(device, non_blocking=True)
        support_labels = batch['support_labels'].to(device)
        query = batch['query_pixel_values'].to(device, non_blocking=True)
        classes = batch['classes']

        with autocast_context(device, amp):
            outputs = model(query, support, support_labels, num_classes=len(classes))

        for t, (frame_output, target) in enumerate(zip(outputs, batch['targets'])):
            image_size = batch['image_sizes'][t]
            detections = model.postprocess_detections(
                frame_output, image_size,
                confidence_threshold=score_threshold,
                max_detections=max_detections,
                nms_iou=nms_iou
            )
            accumulator.add_frame(
                image_size,
                target['boxes_xyxy'].tolist(), target['category_ids'],
                detections['boxes'].tolist(), detections['scores'].tolist(),
                [classes[label] for label in detections['labels'].tolist()]
            )
            memory_tokens.append(int(frame_output['num_memory'][0]))

            if vis_dir is not None and i < num_vis_episodes:
                frame = owl_unnormalize(query[t].float().cpu())
                frame = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                h, w = image_size
                side = max(h, w)
                frame = frame.resize((side, side)).crop((0, 0, w, h))
                names = [category_names.get(c, c) for c in classes]
                colors = [tuple(rng.randint(0, 255, 3).tolist()) for _ in classes]
                visualize_detections(
                    frame, detections, names, colors,
                    output_path=Path(vis_dir) / f"episode_{i}_frame_{t}.jpg",
                    threshold=model.confidence_threshold if vis_threshold is None else vis_threshold
                )
        num_frames += len(outputs)

    elapsed = time.time() - start
    results = accumulator.summarize(per_class=per_class)
    results['avg_memory_tokens'] = float(np.mean(memory_tokens)) if memory_tokens else 0.0
    results['num_episodes'] = len(data_loader)
    results['fps'] = num_frames / max(elapsed, 1e-6)
    return results


def evaluate_model(args):
    """Evaluate model on the test set"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_path is None:
        print("No checkpoint given: evaluating the untrained model (pretrained OWLv2 heads)")
    model = load_model_from_checkpoint(
        args.checkpoint_path, device,
        pretrained_model_name=args.pretrained_model,
        confidence_threshold=args.conf_threshold,
        temporal_confidence_threshold=args.temporal_threshold,
        use_temporal=False if args.no_temporal else None
    )
    model.eval()
    print(f"Model: {model.owl.name_or_path}, temporal fusion: {model.use_temporal}, "
          f"tau={model.temporal_confidence_threshold}, kappa={model.confidence_threshold}")

    test_loader = get_fsvod_loader(
        root_dir=args.data_dir,
        split=args.split,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        image_size=model.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_videos=args.max_videos
    )
    print(f"{args.split}: {len(test_loader.dataset.videos)} videos, {len(test_loader.dataset.all_classes)} classes, "
          f"{len(test_loader)} episodes ({args.n_way}-way {args.k_shot}-shot)")

    category_names = {}
    names_file = Path(args.data_dir) / "categories.json"
    if names_file.exists():
        with open(names_file) as f:
            category_names = json.load(f)

    results = run_evaluation(
        model, test_loader, device,
        amp=not args.no_amp,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        nms_iou=args.nms_iou,
        per_class=True,
        vis_dir=vis_dir,
        num_vis_episodes=args.num_vis_episodes,
        category_names=category_names
    )
    per_class = {category_names.get(k, k): v for k, v in results.pop('per_class', {}).items()}

    print("\nOverall results:")
    for k in ['AP', 'AP50', 'AP75']:
        print(f"{k}: {results[k]:.2f}")
    print(f"frames/s: {results['fps']:.2f}, avg. propagated object tokens: {results['avg_memory_tokens']:.2f}")

    print("\nPer-class AP:")
    for cls_name, ap in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"{cls_name}: {ap:.2f}")

    all_results = {'overall': results, 'per_class': per_class, 'args': vars(args)}
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

    if args.plot_results:
        plt.figure(figsize=(6, 4))
        plt.bar(['AP', 'AP50', 'AP75'], [results['AP'], results['AP50'], results['AP75']])
        plt.title('Overall AP')
        plt.ylabel('AP (%)')
        plt.savefig(output_dir / "overall_map.png", bbox_inches='tight')

        names = sorted(per_class, key=lambda k: -per_class[k])
        plt.figure(figsize=(max(8, 0.3 * len(names)), 5))
        plt.bar(names, [per_class[n] for n in names])
        plt.xticks(rotation=90)
        plt.title('Per-class AP')
        plt.ylabel('AP (%)')
        plt.savefig(output_dir / "per_class_map.png", bbox_inches='tight')

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot Video Object Detection Model")

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--split', type=str, default='test', help="Split to evaluate")
    parser.add_argument('--n_way', type=int, default=5, help="Number of classes per episode")
    parser.add_argument('--k_shot', type=int, default=5, help="Number of examples per class")
    parser.add_argument('--max_frames', type=int, default=0, help="Frames per video (0 = all annotated frames)")
    parser.add_argument('--frame_stride', type=int, default=1, help="Stride over annotated frames")
    parser.add_argument('--max_videos', type=int, default=0, help="Evaluate at most this many videos (0 = all)")

    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Model checkpoint (omit to evaluate the untrained model)")
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Pretrained OWLv2 model (default: from checkpoint, else owlv2-large-patch14-ensemble)")
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help="Detection threshold kappa for visualisation (default 0.98)")
    parser.add_argument('--temporal_threshold', type=float, default=None,
                        help="Threshold tau for temporal propagation (default 0.94)")
    parser.add_argument('--no_temporal', action='store_true', help="Disable temporal fusion (ablation)")

    # Evaluation parameters
    parser.add_argument('--score_threshold', type=float, default=0.0, help="Score threshold for AP evaluation")
    parser.add_argument('--max_detections', type=int, default=100, help="Detections per frame")
    parser.add_argument('--nms_iou', type=float, default=None, help="Class-wise NMS IoU (default: no NMS)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
    parser.add_argument('--output_dir', type=str, default="evaluation_results", help="Output directory")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--no_amp', action='store_true', help="Disable bf16 autocast on GPU")
    parser.add_argument('--seed', type=int, default=42, help="Seed for the evaluation episodes")

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help="Visualize detections")
    parser.add_argument('--num_vis_episodes', type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument('--plot_results', action='store_true', help="Plot AP results")

    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
