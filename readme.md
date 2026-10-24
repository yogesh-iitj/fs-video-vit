# Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection

[![AAAI'26 (PDF)](https://img.shields.io/badge/Paper-AAAI'26-blue)](https://github.com/yogesh-iitj/fs-video-vit/blob/main/kumar_aaai'26.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2602.07801-red)](https://arxiv.org/pdf/2511.13784)


This repository contains the official implementation of the AAAI'26 paper *Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection*.


## Project Structure

```
├── config.py           # Configuration parameters
├── dataloader.py       # Few-shot video dataset and dataloader
├── model.py            # Model architecture
├── losses.py           # Training objective
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── inference.py        # Inference on a video
├── utils.py            # Utility functions
└── tools/              # Dataset preparation scripts
```

## Requirements

```bash
pip install -r requirement.txt
```

## Dataset Format

```
dataset_root/
├── videos/
│   ├── video1/
│   │   ├── frame1.jpg
│   │   └── ...
│   └── ...
├── train_annotations.json
├── val_annotations.json
├── test_annotations.json
└── categories.json             # optional: {category_id: name}
```

Any few-shot video object detection dataset can be used once converted to this format. Train, validation and test splits contain disjoint object categories (base classes for training, novel classes for validation and testing). Each annotation file has the following format:

```json
{
  "video1": {
    "frames": {
      "frame1": {
        "file_name": "video1/frame1.jpg",
        "objects": [{"category_id": "class1", "bbox": [x1, y1, x2, y2]}, ...]
      },
      ...
    }
  },
  ...
}
```

## Training

```bash
python train.py \
  --data_dir /path/to/dataset \
  --pretrained_model google/owlv2-large-patch14-ensemble \
  --n_way 5 --k_shot 5 --max_frames 8 \
  --epochs 20 --episodes_per_epoch 500 \
  --temporal_threshold 0.94 --conf_threshold 0.98 \
  --lr 1e-5 --weight_decay 0.01 --lambda_cls 2.0 --lambda_box 5.0 \
  --output_dir outputs --run_name my_run
```

## Evaluation

```bash
python evaluate.py \
  --data_dir /path/to/dataset --split test \
  --n_way 5 --k_shot 5 \
  --checkpoint_path outputs/my_run/best_model.pt \
  --visualize --plot_results
```

## Inference

```bash
python inference.py \
  --video_path /path/to/video.mp4 \
  --support_folder /path/to/support_images \
  --output_path output_video.mp4 \
  --checkpoint_path outputs/my_run/best_model.pt
```

The support folder has one subfolder of images per class:

```
support_images/
├── class1/
│   ├── image1.jpg
│   └── ...
└── class2/
    └── ...
```


## Citation
If you find this work useful, please consider citing it as:
```bibtex
@inproceedings{kumar2026fsvit,
  title={Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection},
  author={Yogesh Kumar and Anand Mishra},
  booktitle={Association for the Advancement of Artificial Intelligence, AAAI},
  year={2026},
}
```


## Acknowledgements

- This implementation uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for pretrained OWL-ViT models.
