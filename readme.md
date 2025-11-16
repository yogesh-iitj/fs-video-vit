# Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection ([Paper](https://github.com/yogesh-iitj/fs-video-vit/blob/main/kumar_aaai'26.pdf))

This repository contains the implementation of an Object-Aware Temporally Consistent Few-Shot Video Object Detection framework. 


## Project Structure

```
├── config.py                # Configuration parameters
├── dataloader.py            # Few-shot video dataset and dataloader
├── evaluate.py              # Evaluation script for test set
├── inference.py             # Inference script for processing videos
├── model.py                 # Main model architecture
├── train.py                 # Training script
└── utils.py                 # Utility functions
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- transformers
- opencv-python
- Pillow
- numpy
- tqdm
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should be organized as follows:

```
dataset_root/
├── videos/
│   ├── video1/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   ├── video2/
│   │   └── ...
│   └── ...
├── train_annotations.json
├── val_annotations.json
└── test_annotations.json
```

Each annotation file should contain information about videos, frames, and objects in the following format:

```json
{
  "video1": {
    "frames": {
      "frame1": {
        "objects": [
          {
            "category_id": "class1",
            "bbox": [x1, y1, x2, y2]
          },
          ...
        ]
      },
      "frame2": {
        ...
      }
    }
  },
  "video2": {
    ...
  }
}
```

## Training

To train the model:

```bash
python train.py \
  --data_dir /path/to/dataset \
  --n_way 5 \
  --k_shot 5 \
  --max_frames 16 \
  --pretrained_model google/owlv2-large-patch16 \
  --conf_threshold 0.5 \
  --temporal_threshold 0.94 \
  --batch_size 1 \
  --epochs 50 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --lambda_cls 2.0 \
  --lambda_box 5.0 \
  --img_size 640 \
  --output_dir outputs
```

## Evaluation

To evaluate the model on the test set:

```bash
python evaluate.py \
  --data_dir /path/to/dataset \
  --n_way 5 \
  --k_shot 5 \
  --checkpoint_path outputs/best_model.pt \
  --pretrained_model google/owlv2-large-patch16 \
  --conf_threshold 0.5 \
  --temporal_threshold 0.94 \
  --visualize \
  --plot_results
```

## Inference

To run inference on a video:

```bash
python inference.py \
  --video_path /path/to/video.mp4 \
  --support_folder /path/to/support_images \
  --output_path output_video.mp4 \
  --checkpoint_path outputs/best_model.pt \
  --pretrained_model google/owlv2-large-patch16 \
  --conf_threshold 0.5
```

The support folder should be organized as follows, with one subfolder per class:

```
support_images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   └── ...
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

