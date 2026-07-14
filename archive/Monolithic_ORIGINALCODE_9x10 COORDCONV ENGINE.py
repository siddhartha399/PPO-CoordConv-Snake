"""
================================================================================
9x10 COORDCONV ENGINE (V3.1 - T4 DEPLOYMENT READY)
Zero Downsampling | Dilated ResBlocks | Dual-Headed | Channels-Last FP16
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from google.colab import drive

# ── 1. GOOGLE DRIVE COUPLING ──
drive.mount('/content/drive')
print("[OK] Google Drive mounted. Storage secured.")

# ── 2. HARDWARE ACCELERATION TUNING ──
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False

# ── 3. PERSISTENT PATHS ──
CHECKPOINT_DIR = "/content/drive/MyDrive/Snake_AI_PPO/Checkpoints_9x10"
LOG_DIR        = "/content/drive/MyDrive/Snake_AI_PPO/logs_9x10"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class Config:
    # --- 9x10 GOOGLE SNAKE SPECIFIC ---
    GRID_H          = 9
    GRID_W          = 10

    # --- T4 MEMORY-SAFE SWEET SPOT ---
    NUM_ENVS        = 4096
    N_STEPS         = 128
    MINIBATCH_SIZE  = 8192
    EPOCHS          = 4

    LR              = 3e-4
    GAMMA           = 0.99
    GAE_LAMBDA      = 0.95
    CLIP_COEF       = 0.2
    ENT_COEF        = 0.01  # Keeps exploration high in tight spaces
    VF_COEF         = 0.5
    MAX_GRAD_NORM   = 0.5

    REWARD_FOOD     = 1.0
    REWARD_DEATH    = -1.0
    REWARD_STEP     = -0.005
    USE_SHAPING     = True

    SAVE_INTERVAL   = 20
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "ppo_snake_9x10_latest.pt")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "ppo_snake_9x10_best.pt")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ── 4. DILATED RESIDUAL BACKBONE ──
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        # Padding scales with dilation to preserve exact spatial dimensions (9x10)
        padding = dilation
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)

class PPOAgent(nn.Module):
    def __init__(self, h=Config.GRID_H, w=Config.GRID_W):
        super().__init__()
        # 7 Channels: Head, Body, Tail, Food, Danger + Coord X + Coord Y
        self.extractor = nn.Sequential(
            layer_init(nn.Conv2d(7, 64, 3, padding=1)),
            nn.ReLU(),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            # Dilated context blocks to "see" across the 9x10 board without downsampling
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=2)
        )

        # ACTOR HEAD: Compresses to 2 channels before flattening
        self.actor_conv = layer_init(nn.Conv2d(64, 2, 1))
        self.actor_linear = layer_init(nn.Linear(2 * h * w, 4), std=0.01)

        # CRITIC HEAD: Compresses to 1 channel before value estimation
        self.critic_conv = layer_init(nn.Conv2d(64, 1, 1))
        self.critic_hidden = layer_init(nn.Linear(1 * h * w, 256), std=1.0)
        self.critic_out = layer_init(nn.Linear(256, 1), std=1.0)

    def forward_features(self, x):
        features = self.extractor(x)

        # Actor Path
        a = F.relu(self.actor_conv(features))
        a = torch.flatten(a, start_dim=1)  # FIXED: Non-contiguous safe flattening
        logits = self.actor_linear(a)

        # Critic Path
        c = F.relu(self.critic_conv(features))
        c = torch.flatten(c, start_dim=1)  # FIXED: Non-contiguous safe flattening
        v_hidden = F.relu(self.critic_hidden(c))
        value = self.critic_out(v_hidden)

        return logits, value

    def get_value(self, x):
        _, value = self.forward_features(x)
        return value

    def get_action_and_value(self, x, action_mask, action=None):
        logits, value = self.forward_features(x)

        huge_neg = torch.tensor(-1e8, device=Config.DEVICE)
        masked_logits = torch.where(action_mask.bool(), logits, huge_neg)

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

# ── 5. RECTANGULAR (9x10) GPU VECTORIZED ENVIRONMENT ──
class VectorizedSnakeEnv:
    def __init__(self, num_envs, grid_h, grid_w, device):
        self.B = num_envs
        self.H = grid_h
        self.W = grid_w
        self.device = device
        # [y, x] directions: Up, Right, Down, Left
        self.dirs = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], device=device)

        self.grid = torch.zeros((self.B, self.H, self.W), dtype=torch.int32, device=device)
        self.head = torch.zeros((self.B, 2), dtype=torch.long, device=device)
        self.food = torch.zeros((self.B, 2), dtype=torch.long, device=device)
        self.length = torch.full((self.B,), 3, dtype=torch.int32, device=device)
        self.score = torch.zeros(self.B, dtype=torch.float32, device=device)
        self.steps = torch.zeros(self.B, dtype=torch.int32, device=device)

        # COORDCONV STATIC GRIDS (Channels 5 & 6)
        y_coords = torch.linspace(-1.0, 1.0, self.H, device=device).view(1, 1, self.H, 1).expand(self.B, 1, self.H, self.W)
        x_coords = torch.linspace(-1.0, 1.0, self.W, device=device).view(1, 1, 1, self.W).expand(self.B, 1, self.H, self.W)
        self.static_coords = torch.cat([y_coords, x_coords], dim=1)

        self.reset_all()
        self.prev_phi = self.get_phi()

    def reset_all(self):
        self.grid.zero_()
        cy, cx = self.H // 2, self.W // 2
        self.head[:, 0] = cy
        self.head[:, 1] = cx
        self.length.fill_(3)
        self.score.zero_()
        self.steps.zero_()
        self.grid[:, cy, cx] = 3
        self.grid[:, cy, cx+1] = 2
        self.grid[:, cy, cx+2] = 1
        self._place_food(torch.ones(self.B, dtype=torch.bool, device=self.device))

    def reset_envs(self, mask):
        if not mask.any(): return
        cy, cx = self.H // 2, self.W // 2
        self.grid[mask] = 0
        self.head[mask, 0] = cy
        self.head[mask, 1] = cx
        self.length[mask] = 3
        self.score[mask] = 0
        self.steps[mask] = 0
        idx = mask.nonzero(as_tuple=True)[0]
        self.grid[idx, cy, cx] = 3
        self.grid[idx, cy, cx+1] = 2
        self.grid[idx, cy, cx+2] = 1
        self._place_food(mask)

    # --- FIXED: VECTORIZED 100% RELIABLE FOOD PLACEMENT ---
    def _place_food(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return

        empty_mask = (self.grid[idx] == 0)
        has_empty = empty_mask.view(len(idx), -1).any(dim=1)
        valid_idx = idx[has_empty]

        if len(valid_idx) == 0:
            return

        empty_mask_valid = (self.grid[valid_idx] == 0)
        noise = torch.rand(empty_mask_valid.shape, device=self.device)
        valid_noise = torch.where(empty_mask_valid, noise, torch.tensor(-1.0, device=self.device))

        flat_noise = valid_noise.view(len(valid_idx), -1)
        best_idx = flat_noise.argmax(dim=1)

        self.food[valid_idx, 0] = best_idx // self.W
        self.food[valid_idx, 1] = best_idx % self.W

    def get_phi(self):
        dist = torch.abs(self.head[:, 0] - self.food[:, 0]) + torch.abs(self.head[:, 1] - self.food[:, 1])
        return -dist.to(torch.float32)

    def step(self, actions):
        self.steps += 1
        moves = self.dirs[actions]
        new_head = self.head + moves

        out_of_bounds = (new_head[:, 0] < 0) | (new_head[:, 0] >= self.H) | \
                        (new_head[:, 1] < 0) | (new_head[:, 1] >= self.W)

        safe_new_head = new_head.clone()
        safe_new_head[out_of_bounds] = 0

        b_idx = torch.arange(self.B, device=self.device)
        hit_body = (self.grid[b_idx, safe_new_head[:, 0], safe_new_head[:, 1]] > 1) & ~out_of_bounds

        # --- FIXED: GRACEFULLY CATCH WIN CONDITION ---
        won = self.length == (self.H * self.W)
        died = out_of_bounds | hit_body | (self.steps >= (150 + 100 * self.score)) | won
        survived = ~died

        ate_food = survived & (new_head[:, 0] == self.food[:, 0]) & (new_head[:, 1] == self.food[:, 1])

        rewards = torch.full((self.B,), Config.REWARD_STEP, device=self.device)
        rewards[died] = Config.REWARD_DEATH

        self.score[ate_food] += 1
        self.length[ate_food] += 1
        rewards[ate_food] = Config.REWARD_FOOD

        survived_no_food = survived & ~ate_food
        if survived_no_food.any():
            s_idx = survived_no_food.nonzero(as_tuple=True)[0]
            mask_gt_0 = self.grid[s_idx] > 0
            self.grid[s_idx] -= mask_gt_0.to(torch.int32)

        if survived.any():
            s_idx = survived.nonzero(as_tuple=True)[0]
            self.head[s_idx] = new_head[s_idx]
            self.grid[s_idx, self.head[s_idx, 0], self.head[s_idx, 1]] = self.length[s_idx]

        needs_food = ate_food & ~won
        if needs_food.any():
            self._place_food(needs_food)

        if Config.USE_SHAPING:
            new_phi = self.get_phi()
            shape_reward = ((Config.GAMMA * new_phi) - self.prev_phi) * 0.1
            rewards += shape_reward * survived_no_food.to(torch.float32)
            self.prev_phi = new_phi

        info_scores = self.score[died].clone()

        if died.any():
            self.reset_envs(died)
            self.prev_phi[died] = self.get_phi()[died]

        return self.get_states(), rewards, died, info_scores, self.get_legal_masks()

    def get_states(self):
        # Base 5 channels
        states = torch.zeros((self.B, 5, self.H, self.W), dtype=torch.float32, device=self.device)
        b_idx = torch.arange(self.B, device=self.device)

        # Ch 0: Head
        states[b_idx, 0, self.head[:, 0], self.head[:, 1]] = 1.0

        # Ch 1: Body (exclude head)
        body_mask = (self.grid > 1)
        body_mask[b_idx, self.head[:, 0], self.head[:, 1]] = False
        states[:, 1] = body_mask.to(torch.float32)

        # Ch 2: Tail
        states[:, 2] = (self.grid == 1).to(torch.float32)

        # Ch 3: Food
        states[b_idx, 3, self.food[:, 0], self.food[:, 1]] = 1.0

        # Ch 4: Whole Body / Danger Field
        whole_body = (self.grid > 0)
        whole_body[b_idx, self.head[:, 0], self.head[:, 1]] = False
        states[:, 4] = whole_body.to(torch.float32)

        # Concatenate with CoordConv channels (Total 7 Channels)
        full_states = torch.cat([states, self.static_coords], dim=1)

        return full_states.to(memory_format=torch.channels_last)

    def get_legal_masks(self):
        masks = torch.zeros((self.B, 4), dtype=torch.float32, device=self.device)
        b_idx = torch.arange(self.B, device=self.device).unsqueeze(1)
        targets = self.head.unsqueeze(1) + self.dirs.unsqueeze(0)

        valid_bounds = (targets[:, :, 0] >= 0) & (targets[:, :, 0] < self.H) & \
                       (targets[:, :, 1] >= 0) & (targets[:, :, 1] < self.W)

        safe_targets = targets.clone()
        safe_targets[~valid_bounds] = 0

        grid_vals = self.grid[b_idx, safe_targets[:, :, 0], safe_targets[:, :, 1]]
        is_legal = valid_bounds & (grid_vals <= 1)

        masks[is_legal] = 1.0
        no_moves = masks.sum(dim=1) == 0
        masks[no_moves] = 1.0
        return masks

def clean_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# ── 6. COMPILED GAE KERNEL ──
@torch.compile(mode="reduce-overhead")
def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, n_steps):
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

# ── 7. MAIN ENGINE LOOP ──
if __name__ == "__main__":
    writer = SummaryWriter(LOG_DIR)

    print(f"[SYSTEM] Initializing Pure GPU Environment (9x10 CoordConv | {Config.NUM_ENVS} Parallel Channels)...")
    env = VectorizedSnakeEnv(Config.NUM_ENVS, Config.GRID_H, Config.GRID_W, Config.DEVICE)

    agent = PPOAgent().to(Config.DEVICE).to(memory_format=torch.channels_last)
    optimizer = optim.Adam(agent.parameters(), lr=Config.LR, eps=1e-5, foreach=True)
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    update = 0
    best_mean_score = 0.0

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

    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("[System] PyTorch 2.0+ detected. Compiling for reduce-overhead...")
            agent = torch.compile(agent, mode="max-autotune")
            print("    -> Compilation successful!")
        except Exception as e:
            print(f"[!] Compilation failed. Falling back to eager mode safely. Error: {e}")

    print(f"[ENGINE] Hyper-Optimized Infinite GPU Pipeline Active | Batch Size: {Config.N_STEPS * Config.NUM_ENVS:,}")

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

                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
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
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    next_value = agent.get_value(next_obs).flatten()
                advantages, returns = compute_gae(rewards, values, dones, next_value, next_done,
                                                  Config.GAMMA, Config.GAE_LAMBDA, Config.N_STEPS)

            # Memory-safe contiguous reshape
            b_obs = obs.view(-1, 7, Config.GRID_H, Config.GRID_W).contiguous(memory_format=torch.channels_last)
            b_logprobs = logprobs.view(-1)
            b_actions = actions.view(-1)
            b_advantages = advantages.view(-1)
            b_returns = returns.view(-1)
            b_masks = masks.view(-1, 4)

            b_inds = np.arange(Config.N_STEPS * Config.NUM_ENVS)
            clipfracs = []

            # --- PHASE 3 : OPTIMISATION PPO ---
            agent.train()
            for epoch in range(Config.EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), Config.MINIBATCH_SIZE):
                    end = start + Config.MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    with torch.amp.autocast('cuda', dtype=torch.float16):
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

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), Config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()

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
        print("\n[!] Manual Interruption detected. Securing model states to Google Drive...")
        torch.save({
            'update': update,
            'model_state_dict': clean_state_dict(agent.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_score': best_mean_score,
        }, Config.MODEL_PATH)
        print("[!] Emergency save complete. Safe to exit.")

    writer.close()
