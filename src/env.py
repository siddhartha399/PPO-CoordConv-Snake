import torch
from src.config import Config

class VectorizedSnakeEnv:
    """
    A fully GPU-vectorized environment for running thousands of concurrent 
    Snake game instances. Utilizes CoordConv static grids for enhanced spatial awareness.
    """
    def __init__(self, num_envs, grid_h, grid_w, device):
        self.B = num_envs
        self.H = grid_h
        self.W = grid_w
        self.device = device
        
        # Directional Vectors [y, x]: Up, Right, Down, Left
        self.dirs = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], device=device)

        # Core Game State Tensors
        self.grid = torch.zeros((self.B, self.H, self.W), dtype=torch.int32, device=device)
        self.head = torch.zeros((self.B, 2), dtype=torch.long, device=device)
        self.food = torch.zeros((self.B, 2), dtype=torch.long, device=device)
        self.length = torch.full((self.B,), 3, dtype=torch.int32, device=device)
        self.score = torch.zeros(self.B, dtype=torch.float32, device=device)
        self.steps = torch.zeros(self.B, dtype=torch.int32, device=device)

        # CoordConv Static Grids (Channels 5 & 6)
        y_coords = torch.linspace(-1.0, 1.0, self.H, device=device).view(1, 1, self.H, 1).expand(self.B, 1, self.H, self.W)
        x_coords = torch.linspace(-1.0, 1.0, self.W, device=device).view(1, 1, 1, self.W).expand(self.B, 1, self.H, self.W)
        self.static_coords = torch.cat([y_coords, x_coords], dim=1)

        self.reset_all()
        self.prev_phi = self.get_phi()

    def reset_all(self):
        """Global reset for the entire environment batch."""
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
        """Targeted reset for environments that have reached a terminal state."""
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

    def _place_food(self, mask):
        """Vectorized, collision-safe random food placement."""
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
        """Calculates Manhattan distance to food for reward shaping."""
        dist = torch.abs(self.head[:, 0] - self.food[:, 0]) + torch.abs(self.head[:, 1] - self.food[:, 1])
        return -dist.to(torch.float32)

    def step(self, actions):
        """Advances the simulation by one step across all environments concurrently."""
        self.steps += 1
        moves = self.dirs[actions]
        new_head = self.head + moves

        out_of_bounds = (new_head[:, 0] < 0) | (new_head[:, 0] >= self.H) | \
                        (new_head[:, 1] < 0) | (new_head[:, 1] >= self.W)

        safe_new_head = new_head.clone()
        safe_new_head[out_of_bounds] = 0

        b_idx = torch.arange(self.B, device=self.device)
        hit_body = (self.grid[b_idx, safe_new_head[:, 0], safe_new_head[:, 1]] > 1) & ~out_of_bounds

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
        """Generates the 7-channel state representation for the neural network."""
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

        # Ch 4: Danger Field
        whole_body = (self.grid > 0)
        whole_body[b_idx, self.head[:, 0], self.head[:, 1]] = False
        states[:, 4] = whole_body.to(torch.float32)

        # Final Tensor Generation
        full_states = torch.cat([states, self.static_coords], dim=1)
        return full_states.to(memory_format=torch.channels_last)

    def get_legal_masks(self):
        """Calculates valid action space to prevent the agent from instantly killing itself."""
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
