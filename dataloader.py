import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import json
import cv2
from pathlib import Path

class FSVODDataset(Dataset):
    """
    Few-Shot Video Object Detection Dataset
    
    This dataset implements N-way K-shot learning for video object detection.
    Each episode consists of:
    - Support set: N classes with K examples each
    - Query set: A video sequence with frames containing objects of the N classes
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        n_way: int = 5,
        k_shot: int = 5,
        max_frames: int = 16,
        frame_stride: int = 1,
        transform=None,
        target_size: Tuple[int, int] = (640, 640),
        episode_length: int = 100,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            max_frames: Maximum number of frames per video
            frame_stride: Stride for sampling frames
            transform: Image transformations
            target_size: Target size for resizing images (H, W)
            episode_length: Number of episodes in an epoch
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.target_size = target_size
        self.episode_length = episode_length
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load annotation files
        self.annotations = self._load_annotations()
        
        # Get all available classes
        self.all_classes = self._get_all_classes()
        
        # Split classes into base and novel based on split
        self.base_classes, self.novel_classes = self._split_classes()
        
        # For test and val, we use fixed episodes
        if split in ['test', 'val']:
            self.episodes = self._generate_episodes()
        else:
            self.episodes = None
    
    def _load_annotations(self):
        """Load dataset annotations"""
        annotation_file = self.root_dir / f"{self.split}_annotations.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file {annotation_file} not found")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _get_all_classes(self):
        """Get all available classes in the dataset"""
        classes = set()
        
        for video_id, video_info in self.annotations.items():
            for frame_id, frame_info in video_info['frames'].items():
                for obj in frame_info['objects']:
                    classes.add(obj['category_id'])
        
        return sorted(list(classes))
    
    def _split_classes(self):
        """Split classes into base and novel based on split"""
        # For simplicity, we use a deterministic split based on class IDs
        # In practice, this should be defined by the dataset
        all_classes = self.all_classes
        n_classes = len(all_classes)
        
        # 60% for base classes, 20% for validation, 20% for testing
        n_base = int(n_classes * 0.6)
        n_val = int(n_classes * 0.2)
        
        if self.split == 'train':
            # Base classes for training
            return all_classes[:n_base], []
        elif self.split == 'val':
            # Novel classes for validation
            return all_classes[:n_base], all_classes[n_base:n_base+n_val]
        else:  # test
            # Novel classes for testing
            return all_classes[:n_base], all_classes[n_base+n_val:]
    
    def _get_support_examples(self, class_id):
        """Get support examples for a specific class"""
        examples = []
        
        # Find all objects of this class in the dataset
        for video_id, video_info in self.annotations.items():
            for frame_id, frame_info in video_info['frames'].items():
                for obj in frame_info['objects']:
                    if obj['category_id'] == class_id:
                        # Add this frame as a support example
                        examples.append({
                            'video_id': video_id,
                            'frame_id': frame_id,
                            'bbox': obj['bbox']  # [x1, y1, x2, y2]
                        })
        
        # Randomly select k_shot examples
        if len(examples) >= self.k_shot:
            return random.sample(examples, self.k_shot)
        else:
            # If not enough examples, sample with replacement
            return random.choices(examples, k=self.k_shot)
    
    def _get_query_video(self, class_ids):
        """Get a query video containing objects of the selected classes"""
        valid_videos = []
        
        # Find videos containing all selected classes
        for video_id, video_info in self.annotations.items():
            video_classes = set()
            
            for frame_info in video_info['frames'].values():
                for obj in frame_info['objects']:
                    video_classes.add(obj['category_id'])
            
            # Check if this video contains all selected classes
            if all(cls_id in video_classes for cls_id in class_ids):
                valid_videos.append(video_id)
        
        # If no valid videos, relax constraint to contain at least one class
        if not valid_videos:
            valid_videos = []
            for video_id, video_info in self.annotations.items():
                video_classes = set()
                
                for frame_info in video_info['frames'].values():
                    for obj in frame_info['objects']:
                        video_classes.add(obj['category_id'])
                
                # Check if this video contains at least one selected class
                if any(cls_id in video_classes for cls_id in class_ids):
                    valid_videos.append(video_id)
        
        # Randomly select a video
        if valid_videos:
            selected_video_id = random.choice(valid_videos)
            return self._prepare_query_video(selected_video_id, class_ids)
        else:
            # Fallback: just select any video
            selected_video_id = random.choice(list(self.annotations.keys()))
            return self._prepare_query_video(selected_video_id, class_ids)
    
    def _prepare_query_video(self, video_id, class_ids):
        """Prepare query video by selecting appropriate frames"""
        video_info = self.annotations[video_id]
        
        # Sort frames by ID
        frame_ids = sorted(video_info['frames'].keys())
        
        # Select frames with stride
        selected_frames = frame_ids[::self.frame_stride]
        
        # Limit to max_frames
        if len(selected_frames) > self.max_frames:
            if self.split in ['test', 'val']:
                # For test/val, take first max_frames
                selected_frames = selected_frames[:self.max_frames]
            else:
                # For train, randomly select max_frames
                selected_frames = sorted(random.sample(selected_frames, self.max_frames))
        
        # Prepare frame data
        frames_data = []
        for frame_id in selected_frames:
            frame_info = video_info['frames'][frame_id]
            
            # Filter objects by selected classes
            objects = [obj for obj in frame_info['objects'] if obj['category_id'] in class_ids]
            
            frames_data.append({
                'frame_id': frame_id,
                'image_path': os.path.join(self.root_dir, 'videos', video_id, f"{frame_id}.jpg"),
                'objects': objects
            })
        
        return {
            'video_id': video_id,
            'frames': frames_data
        }
    
    def _generate_episodes(self):
        """Generate fixed episodes for validation and testing"""
        episodes = []
        
        # Available classes for this split
        available_classes = self.novel_classes if self.novel_classes else self.base_classes
        
        for _ in range(self.episode_length):
            # Randomly select n_way classes
            if len(available_classes) >= self.n_way:
                episode_classes = random.sample(available_classes, self.n_way)
            else:
                # If not enough classes, sample with replacement
                episode_classes = random.choices(available_classes, k=self.n_way)
            
            # Get support examples for each class
            support_set = {}
            for cls_id in episode_classes:
                support_set[cls_id] = self._get_support_examples(cls_id)
            
            # Get query video
            query_video = self._get_query_video(episode_classes)
            
            episodes.append({
                'classes': episode_classes,
                'support_set': support_set,
                'query_video': query_video
            })
        
        return episodes
    
    def _load_image(self, image_path, bbox=None):
        """Load and preprocess an image, optionally cropping to bbox"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if bbox is not None:
                # Crop to bounding box
                x1, y1, x2, y2 = bbox
                image = image.crop((x1, y1, x2, y2))
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return empty tensor as fallback
            return torch.zeros((3, *self.target_size))
    
    def _prepare_episode(self, idx):
        """Prepare an episode (support set + query video)"""
        if self.split in ['test', 'val'] and self.episodes is not None:
            # Use pre-generated episodes for test/val
            episode = self.episodes[idx % len(self.episodes)]
        else:
            # Generate dynamic episode for training
            # Available classes for this split
            available_classes = self.novel_classes if self.novel_classes else self.base_classes
            
            # Randomly select n_way classes
            if len(available_classes) >= self.n_way:
                episode_classes = random.sample(available_classes, self.n_way)
            else:
                # If not enough classes, sample with replacement
                episode_classes = random.choices(available_classes, k=self.n_way)
            
            # Get support examples for each class
            support_set = {}
            for cls_id in episode_classes:
                support_set[cls_id] = self._get_support_examples(cls_id)
            
            # Get query video
            query_video = self._get_query_video(episode_classes)
            
            episode = {
                'classes': episode_classes,
                'support_set': support_set,
                'query_video': query_video
            }
        
        # Load support images
        support_images = {}
        for cls_id, examples in episode['support_set'].items():
            cls_images = []
            for example in examples:
                image_path = os.path.join(
                    self.root_dir, 'videos', example['video_id'], f"{example['frame_id']}.jpg"
                )
                # Load image and crop to object
                image = self._load_image(image_path, example['bbox'])
                cls_images.append(image)
            support_images[cls_id] = torch.stack(cls_images)
        
        # Load query video frames
        query_frames = []
        frame_annotations = []
        
        for frame_data in episode['query_video']['frames']:
            # Load frame image
            frame = self._load_image(frame_data['image_path'])
            query_frames.append(frame)
            
            # Prepare annotations
            objects = []
            for obj in frame_data['objects']:
                if obj['category_id'] in episode['classes']:
                    objects.append({
                        'category_id': obj['category_id'],
                        'bbox': obj['bbox']  # [x1, y1, x2, y2] format
                    })
            
            frame_annotations.append({
                'frame_id': frame_data['frame_id'],
                'objects': objects
            })
        
        query_frames = torch.stack(query_frames) if query_frames else torch.zeros((0, 3, *self.target_size))
        
        return {
            'support_images': support_images,  # Dict[class_id -> Tensor[K, 3, H, W]]
            'query_frames': query_frames,      # Tensor[T, 3, H, W]
            'frame_annotations': frame_annotations,  # List[Dict] annotations for each frame
            'classes': episode['classes']      # List of class IDs in this episode
        }
    
    def __len__(self):
        """Return the number of episodes in the dataset"""
        return self.episode_length
    
    def __getitem__(self, idx):
        """Get an episode"""
        return self._prepare_episode(idx)


def collate_episodes(batch):
    """
    Custom collate function for episode batches
    
    Args:
        batch: List of episodes from dataset __getitem__
        
    Returns:
        Batched episode data
    """
    # Since each element in batch is already a complete episode,
    # and we process one episode at a time, simply return the first item
    return batch[0]


def get_fsvod_loaders(
    root_dir: str,
    n_way: int = 5,
    k_shot: int = 5,
    max_frames: int = 16,
    batch_size: int = 1,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (640, 640)
):
    """
    Get dataloaders for all splits
    
    Args:
        root_dir: Root directory of the dataset
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        max_frames: Maximum number of frames per video
        batch_size: Batch size (typically 1 for episodes)
        num_workers: Number of workers for dataloader
        target_size: Target size for resizing images (H, W)
        
    Returns:
        Dict containing dataloaders for train, val, and test splits
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FSVODDataset(
        root_dir=root_dir,
        split='train',
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        transform=transform,
        target_size=target_size
    )
    
    val_dataset = FSVODDataset(
        root_dir=root_dir,
        split='val',
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        transform=transform,
        target_size=target_size
    )
    
    test_dataset = FSVODDataset(
        root_dir=root_dir,
        split='test',
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        transform=transform,
        target_size=target_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }