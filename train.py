"""
================================================================================
9x10 COORDCONV ENGINE (V1.0 - T4 DEPLOYMENT READY)
Zero Downsampling | Dilated ResBlocks | Dual-Headed | Channels-Last FP16
================================================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.env import VectorizedSnakeEnv
from src.model import PPOAgent, clean_state_dict
from src.utils import compute_gae, get_autocast_context

def setup_environment():
    """Validates directories and attempts cloud integrations if applicable."""
    # Attempt to mount Google Drive if running in a Colab environment
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[OK] Google Drive mounted. Storage secured.")
    except ImportError:
        print("[INFO] Not running in Google Colab. Utilizing local storage pathways.")

    # Hardware Acceleration Tuning
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False

    # Establish Persistent Paths
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    setup_environment()
    writer = SummaryWriter(Config.LOG_DIR)

    print(f"[SYSTEM] Initializing Pure GPU Environment (9x10 CoordConv | {Config.NUM_ENVS} Parallel Channels)...")
    env = VectorizedSnakeEnv(Config.NUM_ENVS, Config.GRID_H, Config.GRID_W, Config.DEVICE)

    agent = PPOAgent().to(Config.DEVICE).to(memory_format=torch.channels_last)
    optimizer = optim.Adam(agent.parameters(), lr=Config.LR, eps=1e-5, foreach=True)
    if Config.DEVICE.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    global_step = 0
    update = 0
    best_mean_score = 0.0

    # Model Resumption Logic
    if os.path.exists(Config.MODEL_PATH):
        print(f"[+] Found Checkpoint: {Config.MODEL_PATH}")
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
        agent.load_state_dict(clean_state_dict(checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        update = checkpoint['update']
        global_step = update * Config.NUM_ENVS * Config.N_STEPS
        
        if 'best_mean_score' in checkpoint:
            best_mean_score = checkpoint['best_mean_score']
        print(f"    -> Resume Update {update} | Step {global_step:,} | Best Score: {best_mean_score:.2f}")
    else:
        print("[+] No checkpoint found. Starting from scratch on 9x10.")

    # PyTorch 2.0+ Optimization
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("[System] PyTorch 2.0+ detected. Compiling for reduce-overhead...")
            agent = torch.compile(agent, mode="max-autotune")
            print("    -> Compilation successful!")
        except Exception as e:
            print(f"[!] Compilation failed. Falling back to eager mode safely. Error: {e}")

    print(f"[ENGINE] Hyper-Optimized Infinite GPU Pipeline Active | Batch Size: {Config.N_STEPS * Config.NUM_ENVS:,}")

    # Tensor Pre-allocation
    obs = torch.zeros((Config.N_STEPS, Config.NUM_ENVS, 7, Config.GRID_H, Config.GRID_W), device=Config.DEVICE)
    actions = torch.zeros((Config.N_STEPS, Config.NUM_ENVS), device=Config.DEVICE)
    logprobs = torch.zeros((Config.N_STEPS, Config.NUM_ENVS), device=Config.DEVICE)
    rewards = torch.zeros((Config.N_STEPS, Config.NUM_ENVS), device=Config.DEVICE)
    dones = torch.zeros((Config.N_STEPS, Config.NUM_ENVS), device=Config.DEVICE)
    values = torch.zeros((Config.N_STEPS, Config.NUM_ENVS), device=Config.DEVICE)
    masks = torch.zeros((Config.N_STEPS, Config.NUM_ENVS, 4), device=Config.DEVICE)

    next_obs = env.get_states()
    next_masks = env.get_legal_masks()
    next_done = torch.zeros(Config.NUM_ENVS, device=Config.DEVICE)

    try:
        while True:
            update += 1
            gpu_dead_scores = []
            start_time = time.time()

            # --- PHASE 1 : VECTORIZED ROLLOUT ---
            agent.eval()
            for step in range(0, Config.N_STEPS):
                global_step += Config.NUM_ENVS

                obs[step].copy_(next_obs)
                masks[step].copy_(next_masks)
                dones[step].copy_(next_done)

                with torch.no_grad(), get_autocast_context(Config.DEVICE):
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_masks)
                    values[step].copy_(value.flatten())

                actions[step].copy_(action)
                logprobs[step].copy_(logprob)

                next_obs, step_rewards, next_done, info_scores, next_masks = env.step(action)
                rewards[step].copy_(step_rewards)

                if len(info_scores) > 0:
                    gpu_dead_scores.append(info_scores)

            scores_batch = torch.cat(gpu_dead_scores).cpu().tolist() if gpu_dead_scores else []

            # --- PHASE 2 : COMPILED GAE ---
            with torch.no_grad(), get_autocast_context(Config.DEVICE):
                next_value = agent.get_value(next_obs).flatten()
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done,
                    Config.GAMMA, Config.GAE_LAMBDA, Config.N_STEPS
                )

            # Memory-safe contiguous reshape
            b_obs = obs.view(-1, 7, Config.GRID_H, Config.GRID_W).contiguous(memory_format=torch.channels_last)
            b_logprobs = logprobs.view(-1)
            b_actions = actions.view(-1)
            b_advantages = advantages.view(-1)
            b_returns = returns.view(-1)
            b_masks = masks.view(-1, 4)

            b_inds = np.arange(Config.N_STEPS * Config.NUM_ENVS)
            clipfracs = []

            # --- PHASE 3 : PPO OPTIMIZATION ---
            agent.train()
            for epoch in range(Config.EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), Config.MINIBATCH_SIZE):
                    end = start + Config.MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    with get_autocast_context(Config.DEVICE):
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_masks[mb_inds], b_actions.long()[mb_inds]
                        )
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            clipfracs += [((ratio - 1.0).abs() > Config.CLIP_COEF).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.CLIP_COEF, 1 + Config.CLIP_COEF)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - Config.ENT_COEF * entropy_loss + v_loss * Config.VF_COEF

                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(agent.parameters(), Config.MAX_GRAD_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), Config.MAX_GRAD_NORM)
                        optimizer.step()

            # --- PHASE 4 : LOGGING & STORAGE ---
            fps = int((Config.NUM_ENVS * Config.N_STEPS) / (time.time() - start_time))

            if scores_batch:
                mean_score = np.mean(scores_batch)
                max_score = np.max(scores_batch)

                writer.add_scalar("Performance/Mean_Score", mean_score, global_step)
                writer.add_scalar("Performance/Max_Score", max_score, global_step)
                writer.add_scalar("Performance/FPS", fps, global_step)

                print(f"Update: {update} | Step: {global_step:,} | Mean Score: {mean_score:.2f} | Max Score: {max_score} | FPS: {fps:,}")

                if mean_score > best_mean_score and mean_score > 5:
                    best_mean_score = mean_score
                    torch.save({
                        'update': update,
                        'model_state_dict': clean_state_dict(agent.state_dict()),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mean_score': best_mean_score,
                    }, Config.BEST_MODEL_PATH)
                    print(f"  [★] NEW HIGH SCORE! Best model secured (Mean: {mean_score:.2f})")

            if update % Config.SAVE_INTERVAL == 0:
                torch.save({
                    'update': update,
                    'model_state_dict': clean_state_dict(agent.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mean_score': best_mean_score,
                }, Config.MODEL_PATH)
                print(f"  [+] Routine Checkpoint saved at Update {update}")

    except KeyboardInterrupt:
        print("\n[!] Manual Interruption detected. Securing model states...")
        torch.save({
            'update': update,
            'model_state_dict': clean_state_dict(agent.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_score': best_mean_score,
        }, Config.MODEL_PATH)
        print("[!] Emergency save complete. Safe to exit.")

    writer.close()
