"""
Check that the fused (SDPA) OWLv2 encoder attention matches transformers' eager implementation.

Usage:
    python tools/check_fused_attention.py google/owlv2-large-patch14-ensemble /path/to/image.jpg
"""
import sys
import time
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection

sys.path.insert(0, '.')
from model import _use_fused_attention
from utils import owl_preprocess


def timed_embed(owl, x):
    start = time.time()
    feature_map, _ = owl.image_embedder(pixel_values=x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return feature_map, time.time() - start


name = sys.argv[1] if len(sys.argv) > 1 else "google/owlv2-large-patch14-ensemble"
image_path = sys.argv[2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False

owl = Owlv2ForObjectDetection.from_pretrained(name).to(device).eval()
x = owl_preprocess(Image.open(image_path).convert('RGB'), owl.config.vision_config.image_size)[None].to(device)
with torch.no_grad():
    eager, t_eager = timed_embed(owl, x)
    _use_fused_attention(owl.owlv2.vision_model)
    timed_embed(owl, x)  # warm-up
    fused, t_fused = timed_embed(owl, x)
err = (eager - fused).abs().max().item()
print(f"fused attention: max abs diff {err:.2e}, eager {t_eager:.3f}s, fused {t_fused:.3f}s")
assert err < 1e-3, err
