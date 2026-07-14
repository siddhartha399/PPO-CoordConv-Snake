import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from src.config import Config

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes network layers with orthogonal weights to prevent gradient vanishing."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def clean_state_dict(state_dict):
    """Removes PyTorch 2.0 compiler prefixes to ensure safe cross-platform checkpoint loading."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

class ResidualBlock(nn.Module):
    """
    Dilated residual block designed to preserve exact spatial dimensions (9x10)
    while expanding the receptive field without downsampling.
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)

class PPOAgent(nn.Module):
    """
    Dual-headed Actor-Critic agent utilizing a Dilated ResBlock backbone 
    and channels-last memory formatting.
    """
    def __init__(self, h=Config.GRID_H, w=Config.GRID_W):
        super().__init__()
        
        # Extractor: 7 Channels (Head, Body, Tail, Food, Danger, Coord X, Coord Y)
        self.extractor = nn.Sequential(
            layer_init(nn.Conv2d(7, 64, 3, padding=1)),
            nn.ReLU(),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=2)
        )

        # Actor Head: Compresses to 2 channels before flattening for action logits
        self.actor_conv = layer_init(nn.Conv2d(64, 2, 1))
        self.actor_linear = layer_init(nn.Linear(2 * h * w, 4), std=0.01)

        # Critic Head: Compresses to 1 channel for state value estimation
        self.critic_conv = layer_init(nn.Conv2d(64, 1, 1))
        self.critic_hidden = layer_init(nn.Linear(1 * h * w, 256), std=1.0)
        self.critic_out = layer_init(nn.Linear(256, 1), std=1.0)

    def forward_features(self, x):
        features = self.extractor(x)

        # Actor Path
        a = F.relu(self.actor_conv(features))
        a = torch.flatten(a, start_dim=1)  
        logits = self.actor_linear(a)

        # Critic Path
        c = F.relu(self.critic_conv(features))
        c = torch.flatten(c, start_dim=1)  
        v_hidden = F.relu(self.critic_hidden(c))
        value = self.critic_out(v_hidden)

        return logits, value

    def get_value(self, x):
        _, value = self.forward_features(x)
        return value

    def get_action_and_value(self, x, action_mask, action=None):
        logits, value = self.forward_features(x)

        # Masking illegal moves to heavily penalize invalid trajectories
        huge_neg = torch.tensor(-1e8, device=Config.DEVICE)
        masked_logits = torch.where(action_mask.bool(), logits, huge_neg)

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value
