import os
import torch

class Config:
    """
    Hyperparameters and configuration settings for the 9x10 Snake AI PPO Engine.
    """
    # --- GRID SPECIFICATIONS ---
    GRID_H          = 9
    GRID_W          = 10

    # --- HARDWARE & MEMORY OPTIMIZATION ---
    NUM_ENVS        = 4096
    N_STEPS         = 128
    MINIBATCH_SIZE  = 8192
    EPOCHS          = 4

    # --- PPO HYPERPARAMETERS ---
    LR              = 3e-4
    GAMMA           = 0.99
    GAE_LAMBDA      = 0.95
    CLIP_COEF       = 0.2
    ENT_COEF        = 0.01  # Maintains exploration in tight spaces
    VF_COEF         = 0.5
    MAX_GRAD_NORM   = 0.5

    # --- REWARD SHAPING ---
    REWARD_FOOD     = 1.0
    REWARD_DEATH    = -1.0
    REWARD_STEP     = -0.005
    USE_SHAPING     = True

    # --- SYSTEM SETTINGS ---
    SAVE_INTERVAL   = 20
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PERSISTENT STORAGE PATHS ---
    # Detect if running in Google Colab to use Drive paths,
    # otherwise default to local project directories.
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        import google.colab  # noqa: F401
        _is_colab = True
    except ImportError:
        _is_colab = False

    if _is_colab:
        CHECKPOINT_DIR = "/content/drive/MyDrive/Snake_AI_PPO/Checkpoints_9x10"
        LOG_DIR        = "/content/drive/MyDrive/Snake_AI_PPO/logs_9x10"
    else:
        CHECKPOINT_DIR = "./checkpoints"
        LOG_DIR        = "./logs"

    MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "ppo_snake_9x10_latest.pt")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "ppo_snake_9x10_best.pt")
