"""
Configuration file for Few-Shot Video Object Detection
"""

class ModelConfig:
    """Model configuration parameters"""
    # Vision encoder
    PRETRAINED_MODEL = "google/owlv2-large-patch14-ensemble"  # OWLv2 ViT-L/14 (1008px input)
    HIDDEN_SIZE = 1024  # Hidden dimension size (D, from the encoder config)
    NUM_ATTENTION_HEADS = 4  # Number of attention heads in temporal decoder
    
    # Thresholds
    TEMPORAL_THRESHOLD = 0.94  # Threshold for temporal propagation (τ)
    FILTERING_THRESHOLD = 0.98  # Threshold for valid detections (κ, --conf_threshold)
    MAX_MEMORY_TOKENS = 64  # Max object tokens propagated to the next frame
    
    # Decoder parameters
    DROPOUT = 0.0  # Dropout rate in decoder
    
    # Whether to freeze the encoder
    FREEZE_ENCODER = True

class TrainingConfig:
    """Training configuration parameters"""
    # Basic training parameters
    BATCH_SIZE = 1  # Batch size (typically 1 for episode-based few-shot learning)
    NUM_EPOCHS = 50  # Number of training epochs
    
    # Optimizer parameters
    LEARNING_RATE = 1e-5  # Learning rate (pretrained heads)
    LEARNING_RATE_NEW = 1e-4  # Learning rate of the temporal decoder
    LEARNING_RATE_BACKGROUND = 1e-2  # Learning rate of the background logit
    WEIGHT_DECAY = 0.01  # Weight decay
    
    # Scheduler parameters
    LR_WARMUP_EPOCHS = 1  # Number of warmup epochs
    LR_MIN_FACTOR = 0.01  # Minimum learning rate factor
    
    # Loss weighting
    LAMBDA_CLS = 2.0  # Classification loss weight
    LAMBDA_BOX = 5.0  # L1 box loss weight
    LAMBDA_GIOU = 2.0  # GIoU loss weight
    FOCAL_GAMMA = 2.0  # Focal modulation of the classification CE
    
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
    MAX_FRAMES = 8  # Frames per training clip
    FRAME_STRIDE = 1  # Stride for sampling frames
    
    # Image parameters
    IMG_SIZE = 1008  # Model input size (set by the OWLv2 model: 960 for B/16, 1008 for L/14)
    
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
