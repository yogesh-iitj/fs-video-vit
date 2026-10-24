"""
Build a few-shot video object detection benchmark from TAO (Dave et al. 2020) in the
annotation format used by this repository (see readme.md).

FSVOD-500 (Fan et al. 2022) is partly built from the same video sources as TAO; like it,
the categories are split into disjoint train (base) / val / test (novel) sets
(64% / 16% / 20%, the FSVOD-500 ratio of 320/80/100). Videos are kept disjoint across splits
as well: a video containing any test class goes to test, otherwise any val class to val,
otherwise to train, so novel objects never appear (unlabelled) in training videos. The most
frequent categories (e.g. person) are forced into the base set so they do not pull most
videos out of the training split.

Usage:
    python tools/prepare_tao_fsvod.py --tao_root /path/to/TAO --output_dir data/tao_fsvod
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_tao(tao_root: Path, splits):
    videos, images, annotations, categories = {}, {}, [], {}
    for split in splits:
        with open(tao_root / 'annotations' / f'{split}.json') as f:
            data = json.load(f)
        for c in data['categories']:
            categories[c['id']] = c['name']
        vid_names = {v['id']: v['name'] for v in data['videos']}
        for v in data['videos']:
            videos[v['name']] = v
        for im in data['images']:
            images[(split, im['id'])] = dict(im, video_name=vid_names[im['video_id']])
        for a in data['annotations']:
            annotations.append(dict(a, image_key=(split, a['image_id'])))
    return videos, images, annotations, categories


def main(args):
    tao_root = Path(args.tao_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos, images, annotations, categories = load_tao(tao_root, args.tao_splits)
    print(f"TAO: {len(videos)} videos, {len(images)} annotated frames, {len(annotations)} boxes")

    # Per-category statistics
    cat_videos = defaultdict(set)
    cat_support = defaultdict(int)
    for a in annotations:
        x, y, w, h = a['bbox']
        video_name = images[a['image_key']]['video_name']
        cat_videos[a['category_id']].add(video_name)
        if min(w, h) >= args.min_support_size:
            cat_support[a['category_id']] += 1
    eligible = sorted(c for c in cat_videos
                      if len(cat_videos[c]) >= args.min_videos and cat_support[c] >= args.min_instances)
    print(f"{len(eligible)} categories with >= {args.min_videos} videos and >= {args.min_instances} support boxes")

    # Category split
    by_frequency = sorted(eligible, key=lambda c: -len(cat_videos[c]))
    forced_train = by_frequency[:args.force_train_topk]
    rest = [c for c in eligible if c not in forced_train]
    random.Random(args.seed).shuffle(rest)
    n_test = int(round(len(eligible) * args.test_ratio))
    n_val = int(round(len(eligible) * args.val_ratio))
    split_classes = {
        'test': set(rest[:n_test]),
        'val': set(rest[n_test:n_test + n_val]),
        'train': set(rest[n_test + n_val:]) | set(forced_train)
    }

    # Video split (test > val > train)
    video_cats = defaultdict(set)
    for a in annotations:
        if a['category_id'] in cat_videos and a['category_id'] in set(eligible):
            video_cats[images[a['image_key']]['video_name']].add(a['category_id'])
    video_split = {}
    for video_name, cats in video_cats.items():
        for split in ['test', 'val', 'train']:
            if cats & split_classes[split]:
                video_split[video_name] = split
                break

    # Frames per video (all annotated frames, including those without objects of the split)
    frames_by_video = defaultdict(dict)
    for key, im in images.items():
        if im['video_name'] in video_split:
            frame_id = Path(im['file_name']).stem
            frames_by_video[im['video_name']][frame_id] = {'file_name': im['file_name'], 'objects': []}
    for a in annotations:
        im = images[a['image_key']]
        video_name = im['video_name']
        split = video_split.get(video_name)
        if split is None or a['category_id'] not in split_classes[split]:
            continue
        x, y, w, h = a['bbox']
        if w < 1 or h < 1:
            continue
        frames_by_video[video_name][Path(im['file_name']).stem]['objects'].append({
            'category_id': str(a['category_id']),
            'bbox': [x, y, x + w, y + h],
            'track_id': a['track_id']
        })

    # Write annotation files
    info = {}
    for split in ['train', 'val', 'test']:
        split_annotations = {}
        for video_name, split_of_video in sorted(video_split.items()):
            if split_of_video != split:
                continue
            v = videos[video_name]
            split_annotations[video_name] = {
                'width': v['width'], 'height': v['height'],
                'frames': frames_by_video[video_name]
            }
        with open(output_dir / f'{split}_annotations.json', 'w') as f:
            json.dump(split_annotations, f)
        used = {o['category_id'] for v in split_annotations.values() for fr in v['frames'].values()
                for o in fr['objects']}
        info[split] = {
            'num_videos': len(split_annotations),
            'num_frames': sum(len(v['frames']) for v in split_annotations.values()),
            'num_boxes': sum(len(fr['objects']) for v in split_annotations.values() for fr in v['frames'].values()),
            'num_classes': len(used),
            'classes': sorted(categories[int(c)] for c in used)
        }
        print(f"{split}: {info[split]['num_videos']} videos, {info[split]['num_frames']} frames, "
              f"{info[split]['num_boxes']} boxes, {info[split]['num_classes']} classes")

    with open(output_dir / 'categories.json', 'w') as f:
        json.dump({str(c): name for c, name in categories.items()}, f, indent=1)
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(dict(info, args=vars(args)), f, indent=1)

    link = output_dir / 'videos'
    if not link.exists():
        os.symlink(tao_root / 'frames', link)
    print(f"Wrote {output_dir} (videos -> {tao_root / 'frames'})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a few-shot video object detection benchmark from TAO")
    parser.add_argument('--tao_root', type=str, required=True, help="TAO root (annotations/ and frames/)")
    parser.add_argument('--tao_splits', nargs='+', default=['train', 'validation'])
    parser.add_argument('--output_dir', type=str, default='data/tao_fsvod')
    parser.add_argument('--min_videos', type=int, default=3, help="Minimum number of videos per category")
    parser.add_argument('--min_instances', type=int, default=5, help="Minimum number of support-sized boxes")
    parser.add_argument('--min_support_size', type=int, default=16, help="Minimum support box side (pixels)")
    parser.add_argument('--val_ratio', type=float, default=0.16)
    parser.add_argument('--test_ratio', type=float, default=0.20)
    parser.add_argument('--force_train_topk', type=int, default=10,
                        help="Most frequent categories always used as base classes")
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
