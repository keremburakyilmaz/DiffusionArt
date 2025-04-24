import os

# Paths
DATA_ROOT = "./raw_images"
CONTENT_IMAGES_DIR = os.path.join(DATA_ROOT, "content")
STYLE_IMAGES_DIR = os.path.join(DATA_ROOT, "style")
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

# Model Configuration
DIFFUSION_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"  # "cuda" or "cpu"

# Training Configuration
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
CONTENT_STRENGTH = 0.8  # How much content to preserve during generation
GUIDANCE_SCALE = 7.5  # For classifier-free guidance
NUM_INFERENCE_STEPS_TRAINING = 10
NUM_INFERENCE_STEPS_INFERENCE = 50

# FFT Configuration
LOW_FREQ_PERCENT = 0.1
MID_FREQ_PERCENT = 0.4

# Create required directories
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CONTENT_IMAGES_DIR, exist_ok=True)
os.makedirs(STYLE_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)