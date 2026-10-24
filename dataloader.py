import os
import re
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from utils import owl_preprocess

# Episodes contain many small tensors (per-frame targets); passing each through its own file
# descriptor exhausts the open-file limit after a few hundred episodes
torch.multiprocessing.set_sharing_strategy('file_system')


def _natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', str(text))]


class FSVODDataset(Dataset):
    """
    Few-Shot Video Object Detection Dataset

    Each episode consists of:
    - Support set: N classes with K object crops each (from other videos than the query)
    - Query set: a clip of a video, annotated with the objects of those N classes

    The classes of a split are the categories annotated in its annotation file, so base
    (train) and novel (val/test) classes are defined by how the files were built; they
    should be disjoint.

    Training episodes are sampled randomly: a video, a clip of max_frames annotated frames
    containing objects, its classes (at most N) plus negative classes up to N. Evaluation
    episodes are fixed: every video of the split is evaluated, with all of its classes
    (in groups of N) and every annotated frame (or the first max_frames when max_frames > 0).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        n_way: int = 5,
        k_shot: int = 5,
        max_frames: int = 8,
        frame_stride: int = 1,
        image_size: int = 1008,
        episode_length: int = 1000,
        seed: int = 42,
        support_context: float = 0.1,
        min_support_size: int = 16,
        training: Optional[bool] = None,
        max_videos: int = 0
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            max_frames: Frames per training clip; for evaluation, 0 = all annotated frames
            frame_stride: Stride for sampling frames
            image_size: Model input size (960 for OWLv2-B/16, 1008 for OWLv2-L/14)
            episode_length: Number of (random) episodes per epoch for training
            seed: Random seed for the fixed evaluation episodes
            support_context: Context added around support boxes, as a fraction of box size
            min_support_size: Minimum box side (pixels) of a support example
            training: Random episodes + augmentation (default: split == 'train')
            max_videos: Use at most this many videos (0 = all), e.g. for quick validation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.episode_length = episode_length
        self.support_context = support_context
        self.min_support_size = min_support_size
        self.training = (split == 'train') if training is None else training

        self.videos = self._load_annotations(max_videos, seed)
        self._index_classes()

        if self.training:
            self.episodes = None
        else:
            self.episodes = self._generate_episodes(random.Random(seed))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def _load_annotations(self, max_videos: int, seed: int):
        """
        Load annotations: {video_id: {"frames": {frame_id: {"objects": [{"category_id", "bbox" (xyxy)}],
        "file_name" (optional, relative to <root>/videos)}}}}
        """
        annotation_file = self.root_dir / f"{self.split}_annotations.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file {annotation_file} not found")
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        videos = []
        for video_id in sorted(annotations.keys(), key=_natural_key):
            frames = []
            frame_dict = annotations[video_id]['frames']
            for frame_id in sorted(frame_dict.keys(), key=_natural_key):
                frame_info = frame_dict[frame_id]
                file_name = frame_info.get('file_name', os.path.join(video_id, f"{frame_id}.jpg"))
                objects = [{'category_id': str(obj['category_id']), 'bbox': [float(v) for v in obj['bbox']]}
                           for obj in frame_info['objects']]
                frames.append({
                    'frame_id': frame_id,
                    'image_path': str(self.root_dir / 'videos' / file_name),
                    'objects': objects
                })
            if frames:
                videos.append({'video_id': video_id, 'frames': frames})

        if max_videos and len(videos) > max_videos:
            videos = sorted(random.Random(seed).sample(videos, max_videos), key=lambda v: _natural_key(v['video_id']))
        return videos

    def _index_classes(self):
        """Index support instances and videos per class"""
        self.class_instances = {}
        self.class_videos = {}
        for v_idx, video in enumerate(self.videos):
            for f_idx, frame in enumerate(video['frames']):
                for obj in frame['objects']:
                    cls_id = obj['category_id']
                    self.class_videos.setdefault(cls_id, set()).add(v_idx)
                    x1, y1, x2, y2 = obj['bbox']
                    if min(x2 - x1, y2 - y1) >= self.min_support_size:
                        self.class_instances.setdefault(cls_id, []).append((v_idx, f_idx, obj['bbox']))
        # Only classes with at least one usable support example can form episodes
        self.all_classes = sorted(self.class_instances.keys(), key=_natural_key)

    # ------------------------------------------------------------------
    # Episode construction
    # ------------------------------------------------------------------
    def _sample_support(self, cls_id: str, query_video_idx: int, rng: random.Random):
        """Sample K support instances of a class, preferring videos other than the query video"""
        instances = self.class_instances[cls_id]
        others = [inst for inst in instances if inst[0] != query_video_idx]
        pool = others if others else instances
        if len(pool) >= self.k_shot:
            return rng.sample(pool, self.k_shot)
        return [rng.choice(pool) for _ in range(self.k_shot)]

    def _complete_classes(self, positive: List[str], rng: random.Random):
        """Add negative (absent) classes until the episode has N classes"""
        classes = list(positive)
        negatives = [c for c in self.all_classes if c not in classes]
        rng.shuffle(negatives)
        classes += negatives[:max(0, self.n_way - len(classes))]
        rng.shuffle(classes)
        return classes

    def _frame_classes(self, video, frame_indices):
        classes = []
        for f_idx in frame_indices:
            for obj in video['frames'][f_idx]['objects']:
                if obj['category_id'] in self.class_instances and obj['category_id'] not in classes:
                    classes.append(obj['category_id'])
        return classes

    def _sample_train_episode(self, rng: random.Random):
        video_idx = rng.randrange(len(self.videos))
        video = self.videos[video_idx]
        num_frames = len(video['frames'])
        span = (self.max_frames - 1) * self.frame_stride + 1 if self.max_frames > 0 else num_frames
        start = rng.randrange(max(1, num_frames - span + 1))
        frame_indices = list(range(start, min(num_frames, start + span), self.frame_stride))

        positive = self._frame_classes(video, frame_indices)
        if not positive:
            return self._sample_train_episode(rng)
        rng.shuffle(positive)
        classes = self._complete_classes(positive[:self.n_way], rng)
        support = {c: self._sample_support(c, video_idx, rng) for c in classes}
        return {'video_idx': video_idx, 'frame_indices': frame_indices, 'classes': classes, 'support': support}

    def _generate_episodes(self, rng: random.Random):
        """Fixed evaluation episodes covering every video and every annotated class"""
        episodes = []
        for video_idx, video in enumerate(self.videos):
            frame_indices = list(range(0, len(video['frames']), self.frame_stride))
            if self.max_frames > 0:
                frame_indices = frame_indices[:self.max_frames]
            positive = self._frame_classes(video, frame_indices)
            for start in range(0, len(positive), self.n_way):
                classes = self._complete_classes(positive[start:start + self.n_way], rng)
                support = {c: self._sample_support(c, video_idx, rng) for c in classes}
                episodes.append({'video_idx': video_idx, 'frame_indices': frame_indices,
                                 'classes': classes, 'support': support})
        return episodes

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _open_image(path):
        """Open an image; returns None for missing / corrupt files"""
        try:
            with Image.open(path) as image:
                return image.convert('RGB')
        except (OSError, ValueError) as e:
            print(f"Warning: skipping unreadable image {path}: {e}")
            return None

    def _load_support_crop(self, video_idx, frame_idx, bbox, flip: bool):
        image = self._open_image(self.videos[video_idx]['frames'][frame_idx]['image_path'])
        if image is None:
            return None
        x1, y1, x2, y2 = bbox
        pad_w = (x2 - x1) * self.support_context
        pad_h = (y2 - y1) * self.support_context
        crop = image.crop((
            max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h)),
            min(image.width, int(round(x2 + pad_w))), min(image.height, int(round(y2 + pad_h)))
        ))
        if flip:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        return owl_preprocess(crop, self.image_size)

    def _prepare_episode(self, episode, rng: Optional[random.Random]):
        flip_video = self.training and rng.random() < 0.5
        classes = episode['classes']
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Support set
        support_pixel_values, support_labels = [], []
        for cls_id in classes:
            for video_idx, frame_idx, bbox in episode['support'][cls_id]:
                flip = self.training and rng.random() < 0.5
                crop = self._load_support_crop(video_idx, frame_idx, bbox, flip)
                # Unreadable frame: use the first readable instance of the class instead
                for fallback in self.class_instances[cls_id]:
                    if crop is not None:
                        break
                    crop = self._load_support_crop(*fallback, flip)
                support_pixel_values.append(crop)
                support_labels.append(class_to_idx[cls_id])

        # Query frames and targets
        video = self.videos[episode['video_idx']]
        query_pixel_values, targets, frame_ids, image_sizes = [], [], [], []
        for f_idx in episode['frame_indices']:
            frame = video['frames'][f_idx]
            image = self._open_image(frame['image_path'])
            if image is None:
                continue
            width, height = image.size
            if flip_video:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            query_pixel_values.append(owl_preprocess(image, self.image_size))

            boxes, labels, category_ids = [], [], []
            for obj in frame['objects']:
                if obj['category_id'] not in class_to_idx:
                    continue
                x1, y1, x2, y2 = obj['bbox']
                x1, x2 = max(0.0, x1), min(float(width), x2)
                y1, y2 = max(0.0, y1), min(float(height), y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                if flip_video:
                    x1, x2 = width - x2, width - x1
                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_idx[obj['category_id']])
                category_ids.append(obj['category_id'])

            boxes_xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            side = float(max(width, height))
            norm = boxes_xyxy / side
            boxes_cxcywh = torch.stack([(norm[:, 0] + norm[:, 2]) / 2, (norm[:, 1] + norm[:, 3]) / 2,
                                        norm[:, 2] - norm[:, 0], norm[:, 3] - norm[:, 1]], dim=-1)
            targets.append({
                'boxes': boxes_cxcywh,                     # normalised cxcywh (padded square)
                'labels': torch.tensor(labels, dtype=torch.long),
                'boxes_xyxy': boxes_xyxy,                  # pixels in the original frame
                'category_ids': category_ids
            })
            frame_ids.append(frame['frame_id'])
            image_sizes.append((height, width))

        return {
            'support_pixel_values': torch.stack(support_pixel_values),   # [N*K, 3, S, S]
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_pixel_values': (torch.stack(query_pixel_values) if query_pixel_values
                                   else torch.zeros(0, 3, self.image_size, self.image_size)),  # [T, 3, S, S]
            'targets': targets,
            'classes': classes,                                          # category id of each class index
            'video_id': video['video_id'],
            'frame_ids': frame_ids,
            'image_sizes': image_sizes
        }

    def __len__(self):
        return self.episode_length if self.training else len(self.episodes)

    def __getitem__(self, idx):
        if not self.training:
            return self._prepare_episode(self.episodes[idx], None)
        while True:
            rng = random.Random(random.getrandbits(64))
            episode = self._prepare_episode(self._sample_train_episode(rng), rng)
            if len(episode['targets']):
                return episode


def collate_episodes(batch):
    """
    Each element is a complete episode and episodes are processed one at a time,
    so simply return the first item
    """
    return batch[0]


def get_fsvod_loader(
    root_dir: str,
    split: str,
    n_way: int = 5,
    k_shot: int = 5,
    max_frames: int = 8,
    frame_stride: int = 1,
    image_size: int = 1008,
    episode_length: int = 1000,
    num_workers: int = 4,
    seed: int = 42,
    max_videos: int = 0
) -> DataLoader:
    """Dataloader for one split (episodes are not batched: batch size 1)"""
    dataset = FSVODDataset(
        root_dir=root_dir,
        split=split,
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        frame_stride=frame_stride,
        image_size=image_size,
        episode_length=episode_length,
        seed=seed,
        max_videos=max_videos
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        persistent_workers=dataset.training and num_workers > 0
    )
