"""
Configuration file for Few-Shot Video Object Detection
"""

class ModelConfig:
    """Model configuration parameters"""
    # Vision encoder
    PRETRAINED_MODEL = "google/owlv2-large-patch16"  # Use OWL-ViT Large model
    HIDDEN_SIZE = 1024  # Hidden dimension size
    NUM_ATTENTION_HEADS = 4  # Number of attention heads in temporal decoder
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detections
    TEMPORAL_THRESHOLD = 0.94  # Threshold for temporal propagation (τ)
    FILTERING_THRESHOLD = 0.98  # Threshold for filtering detections (κ)
    
    # Decoder parameters
    DROPOUT = 0.1  # Dropout rate in decoder
    
    # Whether to freeze the encoder
    FREEZE_ENCODER = True

class TrainingConfig:
    """Training configuration parameters"""
    # Basic training parameters
    BATCH_SIZE = 1  # Batch size (typically 1 for episode-based few-shot learning)
    NUM_EPOCHS = 50  # Number of training epochs
    
    # Optimizer parameters
    LEARNING_RATE = 1e-5  # Learning rate
    WEIGHT_DECAY = 0.01  # Weight decay
    
    # Scheduler parameters
    LR_WARMUP_EPOCHS = 5  # Number of warmup epochs
    LR_MIN_FACTOR = 0.01  # Minimum learning rate factor
    
    # Loss weighting
    LAMBDA_CLS = 2.0  # Classification loss weight
    LAMBDA_BOX = 5.0  # Box loss weight
    
    # Data augmentation
    USE_AUGMENTATION = True  # Whether to use data augmentation
    
    # Mixed precision training
    USE_AMP = True  # Whether to use automatic mixed precision

class DataConfig:
    """Dataset configuration parameters"""
    # Few-shot parameters
    N_WAY = 5  # Number of classes per episode
    K_SHOT = 5  # Number of examples per class
    
    # Video parameters
    MAX_FRAMES = 16  # Maximum number of frames per video
    FRAME_STRIDE = 1  # Stride for sampling frames
    
    # Image parameters
    IMG_SIZE = 640  # Image size (H, W)
    
    # Dataset paths (to be filled by user)
    DATA_ROOT = "/path/to/dataset"
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "val"
    TEST_SPLIT = "test"

class ExperimentConfig:
    """Experiment configuration parameters"""
    # Random seed for reproducibility
    SEED = 42
    
    # Output directory
    OUTPUT_DIR = "outputs"
    
    # Checkpoint saving frequency
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
    
    # Validation frequency
    VALIDATE_INTERVAL = 1  # Validate every N epochs
    
    # GPU settings
    GPU_ID = 0  # GPU ID to use, -1 for CPU
    
    # Logging
    LOG_INTERVAL = 10  # Log training progress every N batches
    
    # Visualization during training
    VISUALIZE_TRAIN = False  # Whether to visualize training examples
    NUM_VIS_EXAMPLES = 5  # Number of examples to visualize
