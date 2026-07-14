import contextlib
import torch


def get_autocast_context(device):
    """
    Returns the correct AMP autocast context for the given device.
    Uses FP16 on CUDA, or a no-op context manager on CPU.
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.compile(mode="reduce-overhead")
def compute_gae(rewards, values, dones, next_value, next_done,
                gamma, gae_lambda, n_steps):
    """
    Compiled kernel for Generalized Advantage Estimation (GAE).
    Optimized to reduce CPU overhead during the advantage calculation phase.
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - next_done.float()
            nextvalues = next_value.squeeze()
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns
