import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from transformers import Owlv2ForObjectDetection
from typing import List, Tuple, Dict, Optional

from utils import box_cxcywh_to_xyxy


def _fused_attention_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None,
                             output_attentions=False):
    """Owlv2Attention.forward with F.scaled_dot_product_attention (same math, no P x P matrices)"""
    if attention_mask is not None or causal_attention_mask is not None or output_attentions:
        return type(self).forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions)
    bsz, seq_len, _ = hidden_states.shape
    q, k, v = [proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
               for proj in (self.q_proj, self.k_proj, self.v_proj)]
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
    return self.out_proj(out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)), None


def _use_fused_attention(vision_model: nn.Module):
    """transformers' OWLv2 has no SDPA path; the eager one materialises 16 x 5184^2 weights per layer"""
    for module in vision_model.modules():
        if type(module).__name__ == 'Owlv2Attention':
            module.forward = _fused_attention_forward.__get__(module)


class ObjectAwareTemporalDecoder(nn.Module):
    """
    Object-aware temporal fusion (Eq. 3-4 of the paper).

    The current frame's patch embeddings F_t are the queries; the classification
    embeddings of the previous frame's confident detections V^det_{t-1} (Eq. 2)
    are the keys and values:

        A_t = Softmax(F_t (V^det_{t-1})^T / sqrt(D)),    F^_t = F_t + A_t V^det_{t-1}

    Implemented with multi-head cross-attention (4 heads in the paper). V^det lives in
    the classification space (dim D_cls), so keys/values are projected to D. The output
    projection is zero-initialised: at the start of training F^_t = F_t, i.e. the model
    starts from the pretrained OWLv2 image-conditioned detector and learns how much
    temporal context to add.
    """
    def __init__(self, hidden_size: int, memory_size: int, num_attention_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            kdim=memory_size,
            vdim=memory_size,
            batch_first=True
        )
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(self,
                frame_embeddings: torch.Tensor,
                memory: torch.Tensor,
                memory_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_embeddings: Current frame embeddings F_t [B, P, D]
            memory: Previous frame's confident object embeddings V^det_{t-1} [B, M, D_cls]
            memory_padding_mask: True for padded memory slots [B, M]

        Returns:
            enhanced_embeddings: F^_t [B, P, D]
        """
        # Rows without any memory token would produce NaNs; give them a dummy key and zero output
        empty = memory_padding_mask.all(dim=1)
        if empty.any():
            memory_padding_mask = memory_padding_mask.clone()
            memory_padding_mask[empty, 0] = False

        attn_output, _ = self.cross_attention(
            query=frame_embeddings,
            key=memory,
            value=memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False
        )
        attn_output = attn_output.masked_fill(empty[:, None, None], 0.0)
        return frame_embeddings + attn_output


class FewShotVideoObjectDetection(nn.Module):
    """
    Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection.

    Components (paper section "Method"):
      - Language-aligned vision encoder: frozen OWLv2 image encoder -> patch embeddings E in R^{P x D}
      - Support set encoding: highest-objectness patch per support image (Eq. 1), averaged over
        the K shots into a class prototype
      - Object-aware temporal frame decoder (Eq. 2-4)
      - Few-shot detection heads: classification projection W_cls + cosine similarity to the
        prototypes + softmax over N classes and background (Eq. 5-6); box MLP (Eq. 7).
        Both heads are initialised from the pretrained OWLv2 heads.
    """
    def __init__(
        self,
        pretrained_model_name: str = "google/owlv2-large-patch14-ensemble",
        confidence_threshold: float = 0.98,
        temporal_confidence_threshold: float = 0.94,
        num_attention_heads: int = 4,
        max_memory_tokens: int = 64,
        use_temporal: bool = True,
        detach_memory: bool = False,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 8
    ):
        super().__init__()

        self.owl = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name)
        _use_fused_attention(self.owl.owlv2.vision_model)
        config = self.owl.config
        self.image_size = config.vision_config.image_size
        self.hidden_size = config.vision_config.hidden_size          # D
        self.cls_embed_size = config.text_config.hidden_size         # D_cls (output of W_cls)

        self.confidence_threshold = confidence_threshold                    # kappa
        self.temporal_confidence_threshold = temporal_confidence_threshold  # tau
        self.max_memory_tokens = max_memory_tokens
        self.use_temporal = use_temporal
        self.detach_memory = detach_memory
        self.freeze_encoder = freeze_encoder
        self.encoder_chunk_size = encoder_chunk_size

        # Object-aware temporal decoder
        self.temporal_decoder = ObjectAwareTemporalDecoder(
            hidden_size=self.hidden_size,
            memory_size=self.cls_embed_size,
            num_attention_heads=num_attention_heads
        )

        # Few-shot classification head (W_cls, logit shift/scale) and box head reuse OWLv2's
        # pretrained heads; a learnable background logit turns the N cosine similarities
        # into an (N + 1)-way softmax.
        self.cls_head = self.owl.class_head
        self.box_head = self.owl.box_head
        self.background_logit = nn.Parameter(torch.zeros(1))

        # Encoder (and the objectness head, only used to pick support patches) are frozen
        for param in self.owl.owlv2.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.owl.layer_norm.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.owl.objectness_head.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.owl.owlv2.eval()
        return self

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, S, S] (see utils.owl_preprocess)

        Returns:
            image_feats: Patch embeddings [B, P, D]
            feature_map: Patch embeddings on the grid [B, h, w, D]
        """
        feature_maps = []
        grad_context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with grad_context:
            for chunk in pixel_values.split(self.encoder_chunk_size):
                feature_map, _ = self.owl.image_embedder(pixel_values=chunk)
                feature_maps.append(feature_map)
        feature_map = torch.cat(feature_maps)
        b, h, w, d = feature_map.shape
        return feature_map.reshape(b, h * w, d), feature_map

    # ------------------------------------------------------------------
    # Support set encoding
    # ------------------------------------------------------------------
    def compute_prototypes(self, support_pixel_values: torch.Tensor, support_labels: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
        """
        Support feature extraction (Eq. 1) and prototype generation.

        Args:
            support_pixel_values: [N*K, 3, S, S] support images (object crops)
            support_labels: [N*K] episode class index of each support image
            num_classes: N

        Returns:
            prototypes: [N, D] class prototypes z_i (encoder space)
        """
        support_feats, _ = self.encode_images(support_pixel_values)
        objectness = self.owl.objectness_predictor(support_feats)               # [N*K, P]
        best_patch = objectness.argmax(dim=-1)                                  # p*
        z = support_feats[torch.arange(support_feats.size(0), device=support_feats.device), best_patch]
        prototypes = torch.zeros(num_classes, z.size(-1), device=z.device, dtype=z.dtype)
        prototypes.index_add_(0, support_labels, z)
        counts = torch.bincount(support_labels, minlength=num_classes).clamp(min=1).to(z.dtype)
        return prototypes / counts[:, None]

    # ------------------------------------------------------------------
    # Per-frame decoding
    # ------------------------------------------------------------------
    def forward_frame(
        self,
        image_feats: torch.Tensor,
        feature_map: torch.Tensor,
        prototypes: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_feats: Current frame patch embeddings F_t [B, P, D]
            feature_map: Same embeddings on the grid [B, h, w, D] (for the box bias)
            prototypes: Class prototypes [N, D]
            memory: V^det_{t-1} [B, M, D_cls] or None (first frame / nothing confident)
            memory_padding_mask: [B, M]

        Returns:
            dict with
                logits: [B, P, N + 1] (last index = background)
                probs: [B, P, N + 1]
                boxes: [B, P, 4] cxcywh, normalised to the padded square input
                memory, memory_padding_mask: V^det_t for the next frame
                num_memory: [B] number of propagated object tokens
        """
        # Temporal fusion (t > 0)
        if self.use_temporal and memory is not None:
            enhanced_feats = self.temporal_decoder(image_feats, memory, memory_padding_mask)
        else:
            enhanced_feats = image_feats

        # Classification head: V^cls_t = W_cls F^_t (Eq. 5) and cosine similarity (Eq. 6)
        cls_embeds = self.cls_head.dense0(enhanced_feats)                      # [B, P, D_cls]
        proto_embeds = self.cls_head.dense0(prototypes)                         # [N, D_cls]
        similarity = torch.einsum(
            'bpd,nd->bpn',
            F.normalize(cls_embeds, dim=-1, eps=1e-6),
            F.normalize(proto_embeds, dim=-1, eps=1e-6)
        )
        logit_shift = self.cls_head.logit_shift(enhanced_feats)
        logit_scale = self.cls_head.elu(self.cls_head.logit_scale(enhanced_feats)) + 1
        class_logits = (similarity + logit_shift) * logit_scale                 # [B, P, N]
        background = self.background_logit.to(class_logits.dtype).expand(*class_logits.shape[:2], 1)
        logits = torch.cat([class_logits, background], dim=-1).float()
        probs = logits.softmax(dim=-1)

        # Localisation head (Eq. 7)
        boxes = self.owl.box_predictor(enhanced_feats, feature_map).float()

        # V^det_t: classification embeddings of patches with max class probability > tau (Eq. 2)
        next_memory, next_mask, num_memory = self._select_memory(cls_embeds, probs[..., :-1])

        return {
            'logits': logits,
            'probs': probs,
            'boxes': boxes,
            'memory': next_memory,
            'memory_padding_mask': next_mask,
            'num_memory': num_memory
        }

    def _select_memory(self, cls_embeds: torch.Tensor, class_probs: torch.Tensor):
        """Select the object tokens propagated to the next frame (at most max_memory_tokens)"""
        if not self.use_temporal:
            return None, None, torch.zeros(cls_embeds.size(0), dtype=torch.long, device=cls_embeds.device)

        scores = class_probs.max(dim=-1).values.detach()                        # [B, P]
        keep = scores > self.temporal_confidence_threshold
        num_keep = keep.sum(dim=1).clamp(max=self.max_memory_tokens)            # [B]
        max_keep = int(num_keep.max().item())
        if max_keep == 0:
            return None, None, num_keep

        masked_scores = scores.masked_fill(~keep, -1.0)
        top_scores, top_idx = masked_scores.topk(max_keep, dim=1)               # [B, M]
        memory = torch.gather(cls_embeds, 1, top_idx[..., None].expand(-1, -1, cls_embeds.size(-1)))
        padding_mask = top_scores <= self.temporal_confidence_threshold
        if self.detach_memory:
            memory = memory.detach()
        return memory, padding_mask, num_keep

    # ------------------------------------------------------------------
    # Video forward
    # ------------------------------------------------------------------
    def forward(
        self,
        query_pixel_values: torch.Tensor,
        support_pixel_values: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass over one episode (frames are processed sequentially so that the
        object memory of frame t-1 conditions frame t).

        Args:
            query_pixel_values: Target video frames [T, 3, S, S]
            support_pixel_values: Support images [N*K, 3, S, S]
            support_labels: [N*K] episode class indices
            num_classes: N

        Returns:
            List (length T) of per-frame outputs (see forward_frame), each with batch size 1
        """
        prototypes = self.compute_prototypes(support_pixel_values, support_labels, num_classes)
        frame_feats, frame_maps = self.encode_images(query_pixel_values)

        outputs = []
        memory, memory_mask = None, None
        for t in range(frame_feats.size(0)):
            frame_output = self.forward_frame(
                frame_feats[t:t + 1], frame_maps[t:t + 1], prototypes, memory, memory_mask
            )
            memory = frame_output['memory']
            memory_mask = frame_output['memory_padding_mask']
            outputs.append(frame_output)
        return outputs

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def postprocess_detections(
        self,
        frame_output: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],
        confidence_threshold: Optional[float] = None,
        max_detections: int = 100,
        nms_iou: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw outputs of one frame into detections. Each patch is a candidate whose
        label is its most probable (non-background) class; it is a valid detection if that
        probability exceeds the threshold (kappa by default).

        Args:
            frame_output: Output of forward_frame (batch size 1)
            image_size: Original frame size (H, W)
            confidence_threshold: Score threshold (default: kappa). Use a low value for AP evaluation.
            max_detections: Keep at most this many detections (highest scores)
            nms_iou: Optional class-wise NMS IoU threshold

        Returns:
            dict with boxes [D, 4] (xyxy, pixels), scores [D], labels [D] (episode class index)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        probs = frame_output['probs'][0, :, :-1]
        scores, labels = probs.max(dim=-1)
        boxes = box_cxcywh_to_xyxy(frame_output['boxes'][0])

        # Boxes are relative to the padded square image
        height, width = image_size
        boxes = boxes * max(height, width)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

        keep = scores > confidence_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if nms_iou is not None and scores.numel():
            keep = batched_nms(boxes, scores, labels, nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        order = scores.argsort(descending=True)[:max_detections]

        return {
            'boxes': boxes[order].cpu(),
            'scores': scores[order].cpu(),
            'labels': labels[order].cpu()
        }

    # ------------------------------------------------------------------
    # Checkpoints (only trainable weights are stored when the encoder is frozen)
    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable}

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        missing = [k for k in missing if k in trainable]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. Missing: {missing[:10]} Unexpected: {unexpected[:10]}")


def load_model_from_checkpoint(checkpoint_path: Optional[str], device: torch.device, **overrides):
    """
    Build the model from a checkpoint's saved arguments (overridable) and load its weights.
    With checkpoint_path=None the untrained (pretrained OWLv2) model is returned.
    """
    model_args = {}
    state_dict = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved = checkpoint.get('args', {})
        for key, arg in [('pretrained_model_name', 'pretrained_model'),
                         ('confidence_threshold', 'conf_threshold'),
                         ('temporal_confidence_threshold', 'temporal_threshold'),
                         ('max_memory_tokens', 'max_memory_tokens'),
                         ('num_attention_heads', 'num_heads')]:
            if arg in saved:
                model_args[key] = saved[arg]
        if 'no_temporal' in saved:
            model_args['use_temporal'] = not saved['no_temporal']
        state_dict = checkpoint['model']
    model_args.update({k: v for k, v in overrides.items() if v is not None})
    model = FewShotVideoObjectDetection(**model_args)
    if state_dict is not None:
        model.load_trainable_state_dict(state_dict)
    return model.to(device)
