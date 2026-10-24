import os
import math
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

from model import FewShotVideoObjectDetection
from dataloader import get_fsvod_loader
from losses import SetCriterion
from evaluate import run_evaluation, autocast_context
from utils import AverageMeter, set_seed


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device, epoch, args):
    """Train for one epoch"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    meters = {k: AverageMeter() for k in ['loss', 'loss_cls', 'loss_box', 'loss_giou']}
    memory_tokens = AverageMeter()

    end = time.time()
    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", mininterval=30)):
        data_time.update(time.time() - end)

        support = batch['support_pixel_values'].to(device, non_blocking=True)
        support_labels = batch['support_labels'].to(device)
        query = batch['query_pixel_values'].to(device, non_blocking=True)
        targets = batch['targets']

        with autocast_context(device, args.amp):
            outputs = model(query, support, support_labels, num_classes=len(batch['classes']))
        losses = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        losses['loss'].backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
        optimizer.step()
        scheduler.step()

        for k, meter in meters.items():
            meter.update(losses[k].item())
        memory_tokens.update(sum(int(o['num_memory'][0]) for o in outputs) / len(outputs))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] '
                  f'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                  f'Data {data_time.val:.2f} ({data_time.avg:.2f}) '
                  f'Loss {meters["loss"].val:.3f} ({meters["loss"].avg:.3f}) '
                  f'Cls {meters["loss_cls"].avg:.3f} Box {meters["loss_box"].avg:.3f} '
                  f'GIoU {meters["loss_giou"].avg:.3f} Mem {memory_tokens.avg:.1f} '
                  f'BG {model.background_logit.item():.2f} '
                  f'LR {optimizer.param_groups[0]["lr"]:.2e}', flush=True)

    stats = {k: m.avg for k, m in meters.items()}
    stats['avg_memory_tokens'] = memory_tokens.avg
    return stats


def build_optimizer(model, args):
    """
    AdamW; the newly added temporal decoder can use a larger learning rate than the pretrained
    heads. The background logit (a single scalar) gets its own, larger learning rate: support-image
    similarities are much higher than OWL's text-query similarities, so the background level has
    to move by several units from its initial value of 0 early in training.
    """
    new_params, pretrained_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or name == 'background_logit':
            continue
        if name.startswith('temporal_decoder'):
            new_params.append(param)
        else:
            pretrained_params.append(param)
    lr_new = args.lr_new if args.lr_new is not None else args.lr
    return torch.optim.AdamW([
        {'params': pretrained_params, 'lr': args.lr},
        {'params': new_params, 'lr': lr_new},
        {'params': [model.background_logit], 'lr': args.lr_background, 'weight_decay': 0.0}
    ], weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch):
    """Cosine schedule with linear warmup"""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_ap, history, args):
    torch.save({
        'epoch': epoch,
        'model': model.trainable_state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_ap': best_ap,
        'history': history,
        'args': vars(args)
    }, path)


def main(args):
    set_seed(args.seed)

    # Output directory (fixed when resuming so that a re-submitted job continues)
    if args.run_name:
        output_dir = Path(args.output_dir) / args.run_name
    else:
        output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    args.amp = args.amp and device.type == 'cuda'
    print(f"Using device: {device} (bf16 autocast: {args.amp})")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = FewShotVideoObjectDetection(
        pretrained_model_name=args.pretrained_model,
        confidence_threshold=args.conf_threshold,
        temporal_confidence_threshold=args.temporal_threshold,
        num_attention_heads=args.num_heads,
        max_memory_tokens=args.max_memory_tokens,
        use_temporal=not args.no_temporal,
        detach_memory=args.detach_memory,
        freeze_encoder=not args.unfreeze_encoder
    ).to(device)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.pretrained_model}: input {model.image_size}px, D={model.hidden_size}, "
          f"D_cls={model.cls_embed_size}, trainable params {num_trainable / 1e6:.1f}M")

    loader_args = dict(root_dir=args.data_dir, n_way=args.n_way, k_shot=args.k_shot,
                       frame_stride=args.frame_stride, image_size=model.image_size,
                       num_workers=args.num_workers, seed=args.seed)
    train_loader = get_fsvod_loader(split='train', max_frames=args.max_frames,
                                    episode_length=args.episodes_per_epoch, **loader_args)
    val_loader = get_fsvod_loader(split='val', max_frames=args.eval_max_frames,
                                  max_videos=args.val_max_videos, **loader_args)
    print(f"train: {len(train_loader.dataset.videos)} videos / {len(train_loader.dataset.all_classes)} classes, "
          f"val: {len(val_loader.dataset.videos)} videos / {len(val_loader.dataset.all_classes)} classes / "
          f"{len(val_loader)} episodes")

    criterion = SetCriterion(
        lambda_cls=args.lambda_cls,
        lambda_box=args.lambda_box,
        lambda_giou=args.lambda_giou,
        focal_gamma=args.focal_gamma,
        background_weight=args.background_weight
    )
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))

    start_epoch, best_ap, history = 1, -1.0, []
    last_path = output_dir / "last.pt"
    resume_path = args.resume or (str(last_path) if last_path.exists() else None)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_trainable_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_ap = checkpoint['best_ap']
        history = checkpoint['history']
        print(f"Resumed from {resume_path} (epoch {checkpoint['epoch']}, best val AP {best_ap:.2f})")

    eval_kwargs = dict(amp=args.amp, max_detections=args.max_detections, nms_iou=args.nms_iou)

    if start_epoch == 1 and args.eval_before_training:
        val_metrics = run_evaluation(model, val_loader, device, desc="Val (untrained)", **eval_kwargs)
        print(f" * Untrained val AP {val_metrics['AP']:.2f} AP50 {val_metrics['AP50']:.2f} "
              f"AP75 {val_metrics['AP75']:.2f}", flush=True)
        history.append({'epoch': 0, 'val': val_metrics})

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, args)
        record = {'epoch': epoch, 'train': train_metrics}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = run_evaluation(model, val_loader, device, desc="Val", **eval_kwargs)
            record['val'] = val_metrics
            print(f" * Epoch {epoch} val AP {val_metrics['AP']:.2f} AP50 {val_metrics['AP50']:.2f} "
                  f"AP75 {val_metrics['AP75']:.2f} (mem {val_metrics['avg_memory_tokens']:.1f})", flush=True)
            if val_metrics['AP'] > best_ap:
                best_ap = val_metrics['AP']
                history.append(record)
                save_checkpoint(output_dir / "best_model.pt", model, optimizer, scheduler, epoch, best_ap,
                                history, args)
                print(f"Saved best model with val AP: {best_ap:.2f}")
            else:
                history.append(record)
        else:
            history.append(record)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_ap, history, args)
        with open(output_dir / "stats.json", 'w') as f:
            json.dump(history, f, indent=4)

    print(f"Training completed. Best val AP: {best_ap:.2f}")

    # Final evaluation on the test set with the best model
    if args.skip_test:
        return
    print("Evaluating on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location='cpu', weights_only=False)
    model.load_trainable_state_dict(checkpoint['model'])
    test_loader = get_fsvod_loader(split='test', max_frames=args.eval_max_frames, **loader_args)
    test_metrics = run_evaluation(model, test_loader, device, desc="Test", per_class=True, **eval_kwargs)
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test AP {test_metrics['AP']:.2f} AP50 {test_metrics['AP50']:.2f} AP75 {test_metrics['AP75']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Video Object Detection Training")

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--n_way', type=int, default=5, help="Number of classes per episode")
    parser.add_argument('--k_shot', type=int, default=5, help="Number of examples per class")
    parser.add_argument('--max_frames', type=int, default=8, help="Frames per training clip")
    parser.add_argument('--eval_max_frames', type=int, default=0, help="Frames per eval video (0 = all)")
    parser.add_argument('--frame_stride', type=int, default=1, help="Stride over annotated frames")
    parser.add_argument('--episodes_per_epoch', type=int, default=500, help="Training episodes per epoch")
    parser.add_argument('--val_max_videos', type=int, default=0, help="Validate on at most this many videos")

    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default="google/owlv2-large-patch14-ensemble",
                        help="Pretrained OWLv2 model name or path")
    parser.add_argument('--conf_threshold', type=float, default=0.98, help="Detection threshold kappa")
    parser.add_argument('--temporal_threshold', type=float, default=0.94, help="Propagation threshold tau")
    parser.add_argument('--num_heads', type=int, default=4, help="Heads of the temporal cross-attention")
    parser.add_argument('--max_memory_tokens', type=int, default=64, help="Max propagated object tokens")
    parser.add_argument('--no_temporal', action='store_true', help="Disable temporal fusion (ablation)")
    parser.add_argument('--detach_memory', action='store_true', help="No gradient through propagated tokens")
    parser.add_argument('--unfreeze_encoder', action='store_true', help="Fine-tune the OWLv2 encoder")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate (pretrained heads)")
    parser.add_argument('--lr_new', type=float, default=1e-4, help="Learning rate of the temporal decoder")
    parser.add_argument('--lr_background', type=float, default=1e-2, help="Learning rate of the background logit")
    parser.add_argument('--lr_min_factor', type=float, default=0.01, help="Final LR as fraction of initial")
    parser.add_argument('--warmup_epochs', type=float, default=1.0, help="Linear warmup epochs")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--lambda_cls', type=float, default=2.0, help="Classification loss weight")
    parser.add_argument('--lambda_box', type=float, default=5.0, help="L1 box loss weight")
    parser.add_argument('--lambda_giou', type=float, default=2.0, help="GIoU loss weight")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help="Focal modulation of the CE (0 = plain CE)")
    parser.add_argument('--background_weight', type=float, default=1.0, help="CE weight of background patches")
    parser.add_argument('--clip_grad', type=float, default=1.0, help="Gradient clipping (0 = off)")
    parser.add_argument('--no_amp', dest='amp', action='store_false', help="Disable bf16 autocast")

    # Evaluation parameters
    parser.add_argument('--eval_every', type=int, default=1, help="Validate every N epochs")
    parser.add_argument('--eval_before_training', action='store_true', help="Validate the untrained model")
    parser.add_argument('--max_detections', type=int, default=100, help="Detections per frame")
    parser.add_argument('--nms_iou', type=float, default=None, help="Class-wise NMS IoU for evaluation")
    parser.add_argument('--skip_test', action='store_true', help="Do not evaluate on the test split")

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    parser.add_argument('--run_name', type=str, default=None, help="Run sub-directory (enables auto-resume)")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--log_interval', type=int, default=10, help="Log every N episodes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
