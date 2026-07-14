#!/usr/bin/env python3
"""==============================================================================
evaluate.py

Deterministic Policy Evaluation + Telemetry Dashboard Renderer

Loads a trained PPO checkpoint and runs a single deterministic game,
rendering a rich telemetry dashboard frame-by-frame to an MP4 video.

Usage:
    python evaluate.py --checkpoint <path> --output <path> [--max_steps N]
=============================================================================="""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.distributions.categorical import Categorical

from src.config import Config
from src.env import VectorizedSnakeEnv
from src.model import PPOAgent, clean_state_dict

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate a trained PPO Snake agent and render a telemetry dashboard video."
)
parser.add_argument(
    "--checkpoint", "-c",
    type=str,
    default=Config.BEST_MODEL_PATH,
    help="Path to the trained checkpoint .pt file"
)
parser.add_argument(
    "--output", "-o",
    type=str,
    default="./snake_evaluation.mp4",
    help="Output path for the MP4 video"
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Maximum evaluation steps (default: 1000)"
)

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    import imageio
except ImportError as e:
    print("[ERROR] imageio is required for video rendering. Install with: pip install imageio")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = Config.DEVICE

# Image dimensions
GRID_H = Config.GRID_H
GRID_W = Config.GRID_W
CS = 54                    # cell size in pixels
SIDEBAR_W = 480
MIN_H = 680

# ─────────────────────────────────────────────────────────────────────────────
# GRAPHICS: Color Palette
# hexadecimal RGB tuples
# ─────────────────────────────────────────────────────────────────────────────
BG        = (8,  12,  18)
ARENA_BG  = (10, 14,  22)
PANEL_BG  = (15, 21,  31)
PANEL_BDR = (28, 40,  56)
GRID_CLR  = (22, 32,  44)
DIV_CLR   = (24, 36,  50)

HEAD_CLR  = (0,  230, 160)
HEAD_HL   = (140,255, 220)
BODY_CLR  = (0,  148, 104)
TAIL_CLR  = (0,   72,  50)
FOOD_CLR  = (255,  48,  96)
FOOD_GLOW = (140, 18, 46)
FOOD_SPEC = (255, 160, 180)

TXT_MAIN    = (224, 236, 252)
TXT_MUTED   = (82,   98, 116)
TXT_DIM     = (40,   52,  66)
TXT_DANGER  = (220,  52,  52)

ACC_CYAN    = (40,  144, 255)
ACC_GREEN   = (0,   230, 160)
ACC_RED     = (220,  60,  60)
ACC_ORANGE  = (255, 152,  32)

ACT_LABELS  = ["UP  ⇧", "RIGHT ⇨", "DOWN ⇩", "LEFT  ⇦"]
ACT_DELTAS  = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# ─────────────────────────────────────────────────────────────────────────────
# TYPOGRAPHY HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# Try to load system fonts with fallbacks
_B = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_R = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

F_TITLE  = _font(_B, 27)
F_HEADER = _font(_B, 19)
F_BODY   = _font(_R, 14)
F_MICRO  = _font(_R, 11)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def dark(c, n):
    return (max(0, c[0] - n), max(0, c[1] - n), max(0, c[2] - n))


def panel(d, rect, r=8):
    d.rounded_rectangle(rect, radius=r, fill=PANEL_BG, outline=PANEL_BDR, width=1)


def stat(d, x, y, lbl, val, vc=None):
    d.text((x, y),    lbl, fill=TXT_MUTED,   font=F_MICRO)
    d.text((x, y+14), val, fill=vc or TXT_MAIN, font=F_HEADER)


# ─────────────────────────────────────────────────────────────────────────────
# FRAME COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
def build_frame(obs, probs, cval, entropy, action, mask,
                score, length, step, best, grid_h, grid_w):
    """Render a single telemetry dashboard frame from environment tensors."""
    bw = grid_w * CS
    bh = grid_h * CS
    iw = bw + SIDEBAR_W
    ih = max(bh, MIN_H)
    img = Image.new("RGB", (iw, ih), BG)
    d = ImageDraw.Draw(img)

    # --- Arena background ---
    d.rectangle([0, 0, bw, bh], fill=ARENA_BG)
    for x in range(0, bw + 1, CS):
        d.line([(x, 0), (x, bh)], fill=GRID_CLR, width=1)
    for y in range(0, bh + 1, CS):
        d.line([(0, y), (bw, y)], fill=GRID_CLR, width=1)

    # --- Extract channels ---
    h_ch = obs[0, 0].cpu().numpy()
    b_ch = obs[0, 4].cpu().numpy()
    f_ch = obs[0, 3].cpu().numpy()

    # --- Draw cells ---
    for r in range(grid_h):
        for c in range(grid_w):
            px, py = c * CS, r * CS
            rect = [px + 5, py + 5, px + CS - 5, py + CS - 5]
            if h_ch[r, c] > 0:
                d.rounded_rectangle(rect, radius=7, fill=HEAD_CLR)
                d.rounded_rectangle(
                    [rect[0] + 3, rect[1] + 3, rect[2] - 3, rect[3] - 3],
                    radius=4, outline=HEAD_HL, width=1,
                )
                # Direction arrow
                cx2, cy2 = px + CS // 2, py + CS // 2
                dr, dc = ACT_DELTAS[action]
                ax, ay = cx2 + dc * 14, cy2 + dr * 14
                d.line([(cx2 - dc * 5, cy2 - dr * 5), (ax, ay)], fill=dark(HEAD_CLR, 60), width=3)
                pr2, pc2 = -dc, dr
                sz = 5
                d.polygon(
                    [
                        (ax, ay),
                        (ax - dc * sz + pc2 * sz // 2, ay - dr * sz + pr2 * sz // 2),
                        (ax - dc * sz - pc2 * sz // 2, ay - dr * sz - pr2 * sz // 2),
                    ],
                    fill=dark(HEAD_CLR, 60),
                )
            elif b_ch[r, c] > 0:
                d.rounded_rectangle(
                    rect,
                    radius=4,
                    fill=lerp(TAIL_CLR, BODY_CLR, np.clip(float(b_ch[r, c]), 0.3, 1.0)),
                )
            elif f_ch[r, c] > 0:
                m = 9
                d.ellipse([px + m - 3, py + m - 3, px + CS - m + 3, py + CS - m + 3], fill=FOOD_GLOW)
                d.ellipse([px + m, py + m, px + CS - m, py + CS - m], fill=FOOD_CLR)
                d.ellipse([px + m + 3, py + m + 3, px + m + 9, py + m + 9], fill=FOOD_SPEC)

    # --- Arena border ---
    d.rectangle([0, 0, bw - 1, bh - 1], outline=PANEL_BDR, width=2)
    d.line([(bw, 0), (bw, ih)], fill=DIV_CLR, width=2)

    # --- Sidebar layout ---
    sx = bw + 26
    sw = iw - sx - 14
    y = 0

    # Title
    y += 20
    d.text((sx, y), "NEURAL POLICY INFERENCE", fill=TXT_MAIN, font=F_TITLE)
    y += 34
    d.text((sx, y), "▸ PPO · CoordConv · ResNet Architecture", fill=TXT_MUTED, font=F_MICRO)
    y += 14
    d.line([(sx, y), (iw - 14, y)], fill=DIV_CLR, width=1)
    y += 10

    # Environment variables panel
    p = [sx, y, sx + sw, y + 120]
    panel(d, p)
    cy = y + 9
    d.text((sx + 14, cy), "ENVIRONMENT VARIABLES", fill=TXT_MUTED, font=F_MICRO)
    cy += 18
    fill_pct = (length / max(1, grid_h * grid_w)) * 100
    stat(d, sx + 14,  cy, "SCORE",  f"{score:03d}",      ACC_GREEN)
    stat(d, sx + 110, cy, "BEST",   f"{best:03d}",       ACC_CYAN)
    stat(d, sx + 206, cy, "LENGTH", f"{length:03d}",     TXT_MAIN)
    stat(d, sx + 310, cy, "FILL",   f"{fill_pct:4.1f}%", ACC_ORANGE)
    cy += 44
    stat(d, sx + 14,  cy, "STEP",   f"{step:05d}")
    y = p[3] + 12

    # State value estimation panel
    p = [sx, y, sx + sw, y + 126]
    panel(d, p)
    cy = y + 9
    d.text((sx + 14, cy), "STATE VALUE ESTIMATION — V(s)", fill=TXT_MUTED, font=F_MICRO)
    cy += 18
    vc = ACC_GREEN if cval >= 0 else ACC_RED
    d.text((sx + 14, cy), f"V(s) = {'+' if cval >= 0 else ''}{cval:.6f}", fill=vc, font=F_HEADER)
    cy += 26
    d.text((sx + 14, cy), f"Entropy H(π) = {entropy:.4f} nats", fill=ACC_ORANGE, font=F_BODY)
    cy += 22
    bx, bww, bhh = sx + 14, sw - 28, 10
    d.rounded_rectangle([bx, cy, bx + bww, cy + bhh], radius=3, fill=GRID_CLR)
    mid = bx + bww // 2
    cl = max(-1.0, min(1.0, cval))
    end = int(mid + (bww // 2) * cl)
    if cl >= 0:
        d.rounded_rectangle([mid, cy, end, cy + bhh], radius=3, fill=vc)
    else:
        d.rounded_rectangle([end, cy,  mid, cy + bhh], radius=3, fill=vc)
    d.line([(mid, cy - 2), (mid, cy + bhh + 2)], fill=TXT_DIM, width=1)
    ly = cy + bhh + 4
    d.text((bx,         ly), "−1.0", fill=TXT_DIM, font=F_MICRO)
    d.text((mid - 4,    ly), "0",    fill=TXT_DIM, font=F_MICRO)
    d.text((bx + bww - 22, ly), "+1.0", fill=TXT_DIM, font=F_MICRO)
    y = p[3] + 12

    # Policy probability distribution panel
    ph = 30 + len(ACT_LABELS) * 54 + 10
    p = [sx, y, sx + sw, y + ph]
    panel(d, p)
    cy = y + 9
    d.text((sx + 14, cy), "POLICY PROBABILITY DISTRIBUTION — π(a|s)", fill=TXT_MUTED, font=F_MICRO)
    cy += 20
    for i, (lbl, prob) in enumerate(zip(ACT_LABELS, probs)):
        chosen = (i == action)
        legal = bool(mask[i] > 0)
        ry = cy + i * 54
        if chosen:
            d.rounded_rectangle(
                [sx + 10, ry - 3, sx + sw - 10, ry + 44],
                radius=5,
                fill=dark(PANEL_BG, 6),
                outline=HEAD_CLR if legal else TXT_DANGER,
                width=1,
            )
        lc = TXT_DANGER if not legal else (HEAD_CLR if chosen else TXT_MUTED)
        bk = "  [LOCKED]" if not legal else ("  ← ACTIVE" if chosen else "")
        d.text((sx + 18, ry + 3), f"{lbl}{bk}", fill=lc, font=F_BODY)
        rx, ry2, rw, rh = sx + 18, ry + 24, sw - 100, 12
        d.rounded_rectangle([rx, ry2, rx + rw, ry2 + rh], radius=4, fill=GRID_CLR)
        if legal and prob > 0:
            fw = max(1, int(rw * prob))
            d.rounded_rectangle([rx, ry2, rx + fw, ry2 + rh], radius=4,
                                 fill=HEAD_CLR if chosen else ACC_CYAN)
        pc3 = TXT_DANGER if not legal else (TXT_MAIN if chosen else TXT_MUTED)
        d.text((rx + rw + 12, ry2), f"{prob * 100:5.1f}%", fill=pc3, font=F_BODY)

    y = p[3] + 12

    # Footer
    d.line([(sx, ih - 24), (sx + sw, ih - 24)], fill=DIV_CLR, width=1)
    d.text((sx, ih - 14), "▸ PPO-CoordConv-Snake · TELEMETRY ENGINE", fill=TXT_DIM, font=F_MICRO)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🧠  PPO-COORDCONV-SNAKE  —  POLICY EVALUATION DASHBOARD")
    print("=" * 60)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
    print(f"[INFO] Video output will be saved to: {args.output}")
    print(f"[INFO] Grid: {GRID_H}x{GRID_W}  |  Device: {DEVICE}  |  Max Steps: {args.max_steps}")
    print("-" * 60)

    # --- Init agent ---
    agent = PPOAgent().to(DEVICE).to(memory_format=torch.channels_last)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    raw_sd = clean_state_dict(ckpt["model_state_dict"])
    agent.load_state_dict(raw_sd)
    agent.eval()

    best_mean = ckpt.get("best_mean_score", "N/A")
    print(f"[OK]   Checkpoint loaded. Recorded best mean score: {best_mean}")

    # --- Init environment ---
    env = VectorizedSnakeEnv(num_envs=1, grid_h=GRID_H, grid_w=GRID_W, device=DEVICE)
    obs = env.get_states()
    mask = env.get_legal_masks()

    step = 0
    best = 0
    done = False
    video_frames = []

    print(f"[INFO] Starting deterministic policy rollout ...")

    while step < args.max_steps and not done:
        with torch.no_grad():
            logits, value = agent.forward_features(obs)

            # ── DETERMINISTIC ACTION SELECTION (argmax over masked logits) ──
            huge_neg = torch.tensor(-1e8, device=DEVICE)
            masked_logits = torch.where(mask.bool(), logits, huge_neg)
            action = masked_logits.argmax(dim=-1)

            # Dashboard telemetry
            entropy = Categorical(logits=masked_logits).entropy()[0].item()
            probs = F.softmax(masked_logits, dim=-1)[0].cpu().numpy()

        # Step environment
        obs, rewards, died, info_scores, mask = env.step(action)
        step += 1
        done = bool(died[0].item())

        # Score / length handling (env resets on death, so read from info_scores if done)
        if done:
            # Environment resets on death, so retrieve final score from info
            if len(info_scores) > 0:
                current_score = int(info_scores[0].item())
            else:
                # Fallback for robustness (should not occur in normal play)
                current_score = int(env.score[0].item())
            current_length = current_score + 3
        else:
            current_score = int(env.score[0].item())
            current_length = int(env.length[0].item())

        best = max(best, current_score)

        frame = build_frame(
            obs, probs, value[0].item(), entropy, action[0].item(),
            mask[0].cpu().numpy(), current_score, current_length, step, best,
            GRID_H, GRID_W,
        )
        video_frames.append(np.array(frame))

    print(f"[OK]   Rollout completed at step {step}. Final score: {current_score} | Best: {best}")
    print("[INFO] Encoding frames to MP4 ...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    imageio.mimsave(args.output, video_frames, fps=30)

    print(f"[✓]  Video saved successfully to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
