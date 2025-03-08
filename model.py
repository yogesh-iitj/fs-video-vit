import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

class TemporalDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Multi-head attention for temporal processing
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, 
                frame_embeddings: torch.Tensor,
                prev_embeddings: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frame_embeddings: Current frame embeddings [B, P, D]
            prev_embeddings: Previous frame's embeddings (optional) [B, P', D]
            attention_mask: Attention mask for temporal attention [B, P']
            
        Returns:
            enhanced_embeddings: Temporally enhanced embeddings [B, P, D]
        """
        batch_size, num_patches, hidden_dim = frame_embeddings.shape
        
        # If no previous embeddings, use self-attention
        if prev_embeddings is None:
            prev_embeddings = frame_embeddings
            
        # Temporal multi-head attention
        attn_output, _ = self.temporal_attention(
            query=frame_embeddings,
            key=prev_embeddings,
            value=prev_embeddings,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        
        # Residual connection and layer norm
        frame_embeddings = self.layer_norm1(frame_embeddings + attn_output)
        
        # Feedforward network
        ffn_output = self.ffn(frame_embeddings)
        enhanced_embeddings = self.layer_norm2(frame_embeddings + ffn_output)
        
        return enhanced_embeddings


class FewShotVideoObjectDetection(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "google/owlv2-base-patch16-ensemble",
        confidence_threshold: float = 0.7,
        temporal_confidence_threshold: float = 0.85,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # Initialize OWL-ViT processor and model
        self.processor = Owlv2Processor.from_pretrained(pretrained_model_name)
        self.owl_vit = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name)
        
        # Get model config values if not provided
        if hidden_size is None:
            hidden_size = self.owl_vit.config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = self.owl_vit.config.num_attention_heads
            
        # Add temporal decoder for frame-to-frame consistency
        self.temporal_decoder = TemporalDecoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads
        )
        
        # Few-shot classification head
        self.classification_head = nn.Linear(hidden_size, hidden_size)
        
        # Box regression head (reuse OWL-ViT's box head)
        self.box_head = self.owl_vit.box_head
        
        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.temporal_confidence_threshold = temporal_confidence_threshold
        
        # Freeze encoder parameters if specified
        if freeze_encoder:
            for param in self.owl_vit.owlv2.vision_model.parameters():
                param.requires_grad = False
    
    def extract_features(self, images):
        """Extract features from images using OWL-ViT encoder"""
        inputs = self.processor(images=images, return_tensors="pt").to(next(self.parameters()).device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.owl_vit.owlv2.vision_model(
                inputs.pixel_values,
                return_dict=True
            ).last_hidden_state
            
        return image_features
    
    def process_support_set(self, support_images: List[Dict[str, List]]):
        """
        Process support set to create class prototypes
        
        Args:
            support_images: Dict of support images organized by class
                {class_id: [image1, image2, ...], ...}
        
        Returns:
            class_prototypes: Dict mapping class_id to class prototype embeddings
        """
        class_prototypes = {}
        
        for class_id, images in support_images.items():
            class_embeddings = []
            
            for image in images:
                # Extract features
                features = self.extract_features(image)
                
                # Get objectness scores
                objectness_scores = self.owl_vit.class_head(features, is_training=False).logits.mean(dim=-1)
                
                # Select patch with highest objectness score
                max_objectness_idx = objectness_scores.argmax(dim=1)
                selected_embedding = features[torch.arange(features.size(0)), max_objectness_idx]
                
                class_embeddings.append(selected_embedding)
            
            # Create class prototype by averaging
            class_prototype = torch.cat(class_embeddings).mean(dim=0)
            class_prototypes[class_id] = F.normalize(class_prototype, p=2, dim=-1)
        
        return class_prototypes
    
    def forward_frame(
        self,
        frame_features: torch.Tensor,
        class_prototypes: Dict[str, torch.Tensor],
        prev_object_features: Optional[torch.Tensor] = None,
        prev_object_mask: Optional[torch.Tensor] = None
    ):
        """
        Process a single frame with temporal context
        
        Args:
            frame_features: Current frame feature embeddings [B, P, D]
            class_prototypes: Dict mapping class_id to prototype embeddings
            prev_object_features: Previous frame's object features (optional)
            prev_object_mask: Previous frame's object mask (optional)
            
        Returns:
            dict containing:
                - cls_scores: Classification scores for each patch and class
                - boxes: Predicted bounding boxes
                - object_features: High-confidence object features for next frame
                - object_mask: Mask for high-confidence object features
        """
        batch_size, num_patches, hidden_dim = frame_features.shape
        
        # Apply temporal decoder if we have previous frame information
        if prev_object_features is not None and prev_object_mask is not None:
            enhanced_features = self.temporal_decoder(
                frame_embeddings=frame_features,
                prev_embeddings=prev_object_features,
                attention_mask=prev_object_mask
            )
        else:
            enhanced_features = frame_features
        
        # Classification head
        cls_features = self.classification_head(enhanced_features)
        cls_features = F.normalize(cls_features, p=2, dim=-1)
        
        # Compute similarity with class prototypes
        cls_scores = {}
        for class_id, prototype in class_prototypes.items():
            # Compute cosine similarity
            similarity = torch.matmul(cls_features, prototype.unsqueeze(-1)).squeeze(-1)
            cls_scores[class_id] = similarity
        
        # Box prediction head
        boxes = self.box_head(enhanced_features)
        
        # Select high-confidence object features for temporal propagation
        # Concatenate classification scores for all classes
        all_cls_scores = torch.stack(list(cls_scores.values()), dim=-1)
        max_scores, _ = all_cls_scores.max(dim=-1)
        
        # Create mask for high-confidence objects
        confidence_mask = max_scores > self.temporal_confidence_threshold
        
        # Select corresponding features
        selected_indices = confidence_mask.nonzero(as_tuple=True)
        if selected_indices[0].size(0) > 0:
            object_features = enhanced_features[selected_indices]
            object_mask = torch.zeros_like(confidence_mask)
            object_mask[selected_indices] = True
        else:
            # If no high-confidence detections, use all features
            object_features = enhanced_features
            object_mask = torch.ones_like(confidence_mask)
        
        return {
            "cls_scores": cls_scores,
            "boxes": boxes,
            "object_features": object_features,
            "object_mask": object_mask
        }
    
    def forward(
        self,
        frames: List[torch.Tensor],
        support_images: Dict[str, List],
        return_all_frames: bool = True
    ):
        """
        Forward pass for video object detection
        
        Args:
            frames: List of video frames [T, B, C, H, W]
            support_images: Dict of support images by class
            return_all_frames: Whether to return predictions for all frames
            
        Returns:
            Dict containing detection results for all frames
        """
        # Process support set to get class prototypes
        class_prototypes = self.process_support_set(support_images)
        
        # Process frames sequentially
        all_frame_outputs = []
        prev_object_features = None
        prev_object_mask = None
        
        for frame in frames:
            # Extract features
            frame_features = self.extract_features(frame)
            
            # Process frame with temporal context
            frame_output = self.forward_frame(
                frame_features=frame_features,
                class_prototypes=class_prototypes,
                prev_object_features=prev_object_features,
                prev_object_mask=prev_object_mask
            )
            
            # Update previous object features for next frame
            prev_object_features = frame_output["object_features"]
            prev_object_mask = frame_output["object_mask"]
            
            if return_all_frames:
                all_frame_outputs.append(frame_output)
        
        if return_all_frames:
            return all_frame_outputs
        else:
            # Return only the last frame output
            return frame_output
    
    def postprocess_detections(
        self,
        frame_outputs: Dict,
        image_size: Tuple[int, int],
        confidence_threshold: Optional[float] = None
    ):
        """
        Convert raw model outputs to detection format
        
        Args:
            frame_outputs: Raw model outputs
            image_size: Original image size (H, W)
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Dict containing processed detections (boxes, scores, labels)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        cls_scores = frame_outputs["cls_scores"]
        raw_boxes = frame_outputs["boxes"]
        
        # Combine class scores
        combined_scores = torch.stack(list(cls_scores.values()), dim=-1)
        max_scores, class_indices = combined_scores.max(dim=-1)
        
        # Filter by confidence threshold
        mask = max_scores > confidence_threshold
        filtered_scores = max_scores[mask]
        filtered_class_indices = class_indices[mask]
        filtered_boxes = raw_boxes[mask]
        
        # Convert class indices to class IDs
        class_id_list = list(cls_scores.keys())
        filtered_class_ids = [class_id_list[idx] for idx in filtered_class_indices.cpu().numpy()]
        
        # Scale boxes to image size
        height, width = image_size
        scaled_boxes = []
        for box in filtered_boxes:
            x_center, y_center, w, h = box.cpu().numpy()
            x1 = (x_center - w/2) * width
            y1 = (y_center - h/2) * height
            x2 = (x_center + w/2) * width
            y2 = (y_center + h/2) * height
            scaled_boxes.append([x1, y1, x2, y2])
        
        return {
            "boxes": scaled_boxes,
            "scores": filtered_scores.cpu().numpy().tolist(),
            "labels": filtered_class_ids
        }
    
    def predict_video(
        self,
        video_frames: List,
        support_images: Dict[str, List],
        confidence_threshold: Optional[float] = None
    ):
        """
        End-to-end video prediction
        
        Args:
            video_frames: List of video frames
            support_images: Dict of support images by class
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            List of detection results for each frame
        """
        device = next(self.parameters()).device
        
        # Process frames in batches
        frame_outputs = self.forward(
            frames=video_frames,
            support_images=support_images,
            return_all_frames=True
        )
        
        # Postprocess for each frame
        image_size = video_frames[0].size[-2:]  # Height, Width
        all_detections = []
        
        for frame_idx, frame_output in enumerate(frame_outputs):
            detections = self.postprocess_detections(
                frame_output,
                image_size,
                confidence_threshold
            )
            all_detections.append(detections)
        
        return all_detections