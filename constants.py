# Default model path if not specified in config
DEFAULT_MODEL = "meta-llama/Llama-2-7b"  # Requires Hugging Face authentication

# Default device if not specified in config
DEFAULT_DEVICE = "cuda"  # Will fall back to "cpu" if CUDA not available

# Dataset settings
DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "train"

# Training settings
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 128
