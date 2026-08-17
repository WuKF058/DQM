from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict as TDict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import pandas as pd


# Utility Functions

def moving_average(x, window=10):
    """Calculates the moving average of a 1D array to smooth training curves."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def export_history_csv(history: dict, csv_path: str, add_time_index: bool = True):
    """Exports the rollout history dictionary to a CSV file."""
    if not isinstance(history, dict) or len(history) == 0:
        raise ValueError("The history dictionary is empty or invalid.")

    lengths = [len(v) for v in history.values() if isinstance(v, list)]
    if len(lengths) == 0:
        raise ValueError("The history dictionary contains no list fields.")
    T = min(lengths)

    data = {}
    if add_time_index:
        data["t"] = list(range(T))

    for k, v in history.items():
        if isinstance(v, list):
            data[k] = v[:T]

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df


def export_training_log_csv(log: dict, csv_path: str):
    """Exports the training loop logs (losses, returns, etc.) to a CSV file."""
    lengths = [len(v) for v in log.values() if isinstance(v, list)]
    T = min(lengths) if lengths else 0
    if T == 0:
        raise ValueError("The training log contains no list data.")

    df = pd.DataFrame({k: v[:T] for k, v in log.items() if isinstance(v, list)})
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df

# Nash Bargaining Style Team Reward

def _softplus(x: float) -> float:
    """Numerically stable softplus function to ensure positive utilities."""
    if x > 50.0:
        return x
    if x < -50.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _nash_team_reward(mg_r: float, dg_r: float, cfg) -> float:
    """
    Computes a Nash Bargaining objective-based team reward.
    Encourages cooperation by maximizing the product of individual utilities.
    """
    u_mg = _softplus((mg_r - cfg.nash_ref_mg) / max(cfg.nash_scale_mg, 1e-12))
    u_dg = _softplus((dg_r - cfg.nash_ref_dg) / max(cfg.nash_scale_dg, 1e-12))
    return math.log(u_mg + cfg.nash_eps) + math.log(u_dg + cfg.nash_eps)


# Helpers: Observation Flattening

def _to_numpy(x: Any) -> np.ndarray:
    """Safely converts generic array-like objects or tensors to NumPy arrays."""
    if isinstance(x, np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_mg_obs(mg_obs: TDict[str, np.ndarray]) -> np.ndarray:
    """Flattens a dictionary-based Microgrid observation into a 1D vector."""
    if not isinstance(mg_obs, dict):
        raise TypeError(f"Expected dictionary observation for MicrogridEnv, got: {type(mg_obs)}")
    keys = sorted(mg_obs.keys())
    parts: List[np.ndarray] = []
    for k in keys:
        v = _to_numpy(mg_obs[k]).reshape(-1)
        parts.append(v.astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def ensure_1d(x: Any, dtype=np.float32) -> np.ndarray:
    """Ensures the input is reshaped to a 1D NumPy array."""
    return _to_numpy(x).reshape(-1).astype(dtype)


def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely casts a variable to float, falling back to a default value on error."""
    try:
        return float(x)
    except Exception:
        return float(default)

# Actor and Critic Networks

class MLP(nn.Module):
    """Standard Multi-Layer Perceptron used as the backbone for policy/value networks."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (256, 256), activation=nn.Tanh):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentCentralCritic(nn.Module):
    """
    Centralized Critic Network mapping the global state to a state-value estimation.
    Uses an encoder-decoder architecture with a latent bottleneck.
    """

    def __init__(self, global_dim: int, latent_dim: int = 4, hidden=(256, 256)):
        super().__init__()
        hidden = tuple(hidden)
        if len(hidden) < 1:
            raise ValueError("Hidden layers configuration must contain at least one size.")

        self.encoder = MLP(global_dim, hidden=hidden[:-1] if len(hidden) > 1 else (), activation=nn.Tanh)
        enc_out_dim = hidden[-2] if len(hidden) > 1 else global_dim

        self.latent = nn.Sequential(
            nn.Linear(enc_out_dim, latent_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden[-1]),
            nn.Tanh(),
            nn.Linear(hidden[-1], 1),
        )

    def encode(self, s_global: torch.Tensor) -> torch.Tensor:
        """Extracts the latent representation from the global state."""
        h = self.encoder(s_global)
        return self.latent(h)

    def forward(self, s_global: torch.Tensor) -> torch.Tensor:
        """Predicts the state value V(s)."""
        z = self.encode(s_global)
        return self.value_head(z).squeeze(-1)


class ClassicalKernelRegularizer:
    """
    Radial Basis Function (RBF/Gaussian) kernel used for critic smoothness regularization.
    Replaces the previous quantum kernel implementation to ensure purely classical,
    highly efficient execution.
    """

    def __init__(self, feature_dim: int, gamma: float = 1.0):
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.enabled = True
        print(">>> Classical Kernel (RBF) has been activated for PPO! <<<")

    def matrix(self, z: torch.Tensor) -> torch.Tensor:
        # Detach 'z' to emulate the identical non-differentiable behavior
        # of the original quantum kernel version. This crucially prevents PyTorch
        # from triggering complex CUDA dynamic tracking issues (e.g., libcupti).
        z_det = z.detach()
        # Compute squared Euclidean distances between latent features
        dist_sq = torch.cdist(z_det, z_det, p=2.0) ** 2
        # Apply the RBF Gaussian kernel function
        k = torch.exp(-self.gamma * dist_sq)
        return k


class CategoricalActor(nn.Module):
    """
    Discrete action space Actor Network outputting categorical distributions.
    Capable of handling multi-dimensional discrete actions using multiple linear heads.
    """

    def __init__(self, obs_dim: int, nvec: Tuple[int, ...] = (3, 3), hidden=(256, 256)):
        super().__init__()
        self.nvec = tuple(int(n) for n in nvec)
        self.backbone = MLP(obs_dim, hidden=hidden, activation=nn.Tanh)
        self.heads = nn.ModuleList([nn.Linear(hidden[-1], n) for n in self.nvec])

    def _dist(self, obs: torch.Tensor) -> List[torch.distributions.Categorical]:
        """Generates categorical distributions for each action dimension."""
        h = self.backbone(obs)
        return [torch.distributions.Categorical(logits=head(h)) for head in self.heads]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples an action and its corresponding log-probability."""
        dists = self._dist(obs)
        actions = []
        logps = []
        for dist in dists:
            if deterministic:
                a = torch.argmax(dist.probs, dim=-1)
            else:
                a = dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
        action = torch.stack(actions, dim=-1).long()
        logp = torch.stack(logps, dim=-1).sum(dim=-1)
        return action, logp

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluates the log-probability of a specific action given an observation."""
        dists = self._dist(obs)
        logps = []
        for i, dist in enumerate(dists):
            logps.append(dist.log_prob(action[:, i]))
        return torch.stack(logps, dim=-1).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Calculates the entropy of the action distributions to encourage exploration."""
        dists = self._dist(obs)
        ents = [d.entropy() for d in dists]
        return torch.stack(ents, dim=-1).sum(dim=-1)


# Generalized Advantage Estimation (GAE)

def compute_gae(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, last_value: float,
                gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes GAE (Advantages) and TD(lambda) Returns for PPO optimization.
    Returns:
        adv (np.ndarray): Advantage estimations.
        ret (np.ndarray): Calculated target returns (Advantages + Values).
    """
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = float(last_value) if t == T - 1 else float(values[t + 1])

        # Temporal Difference (TD) Error
        delta = float(rewards[t]) + gamma * next_value * next_non_terminal - float(values[t])

        # Exponential moving average of TD errors
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae

    ret = adv + values.astype(np.float32)
    return adv.astype(np.float32), ret.astype(np.float32)

# Plotting Helpers (Logged per Update)

def plot_training_curves(log: TDict[str, List[float]], save_path: str = "training_curves.png"):
    """Visualizes training returns, actor/critic losses, and entropy over time."""
    raw = np.array(log.get("episode_return_team", []), dtype=np.float32)
    smooth = moving_average(raw, window=10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Plot 1: Episode Returns
    if len(raw) > 0: axes[0].plot(np.arange(len(raw)), raw, alpha=0.35, label="Raw Return")
    if len(smooth) > 0: axes[0].plot(np.arange(len(smooth)) + 9, smooth, label="MA(10) Smoothed")
    axes[0].set_title("Episode Team Return (Per Update Horizon)")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Training Losses
    axes[1].plot(np.arange(len(log.get("mg_actor_loss", []))), log.get("mg_actor_loss", []),
                 label="MG Actor Loss (Avg)")
    for k in ("mg_actor_loss_mg1", "mg_actor_loss_mg2", "mg_actor_loss_mg3", "mg_actor_loss_mg4"):
        if k in log and len(log[k]) > 0:
            axes[1].plot(np.arange(len(log[k])), log[k], label=k)

    axes[1].plot(np.arange(len(log.get("dg_actor_loss", []))), log.get("dg_actor_loss", []), label="DG Actor Loss")
    axes[1].plot(np.arange(len(log.get("critic_loss", []))), log.get("critic_loss", []), label="Critic Total Loss",
                 linewidth=2)

    if len(log.get("critic_td_loss", [])) > 0:
        axes[1].plot(np.arange(len(log.get("critic_td_loss", []))), log.get("critic_td_loss", []),
                     label="Critic TD Loss", linestyle='--')
    if len(log.get("critic_kernel_loss", [])) > 0:
        axes[1].plot(np.arange(len(log.get("critic_kernel_loss", []))), log.get("critic_kernel_loss", []),
                     label="Critic Kernel Loss", linestyle=':')

    axes[1].set_title("Training Signals (Losses Per Update)")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    axes[1].legend()

    # Plot 3: Entropy and Timing
    axes[2].plot(np.arange(len(log.get("entropy_mg", []))), log.get("entropy_mg", []), label="MG Entropy (Avg)")
    for k in ("entropy_mg1", "entropy_mg2", "entropy_mg3", "entropy_mg4"):
        if k in log and len(log[k]) > 0:
            axes[2].plot(np.arange(len(log[k])), log[k], label=k)

    axes[2].plot(np.arange(len(log.get("entropy_dg", []))), log.get("entropy_dg", []), label="DG Entropy")
    axes[2].plot(np.arange(len(log.get("update_time_sec", []))), log.get("update_time_sec", []),
                 label="Update Time (s)")
    axes[2].set_title("Exploration Entropy / Update Execution Time")
    axes[2].set_xlabel("Update Step")
    axes[2].set_ylabel("Value")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_grid_results(history: TDict[str, List[float]], save_path: str | None = None, show: bool = False):
    """Plots operational metrics of the Distribution Grid (e.g., pricing, grid power)."""
    t = np.arange(len(history.get("dg_reward", [])))
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(t, history.get("buy_price", []), label="Buy Price")
    axes[0].plot(t, history.get("sell_price", []), label="Sell Price")
    axes[0].set_ylabel("€/kWh")
    axes[0].set_title("Distribution Grid Pricing Dynamic")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, history.get("grid_power", []))
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("kW")
    axes[1].set_title("PCC Total Power Exchange (+Buy / -Sell)")
    axes[1].grid(True)

    axes[2].plot(t, history.get("upstream_power", np.zeros_like(t)))
    axes[2].set_ylabel("kW")
    axes[2].set_title("Upstream Purchase Power")
    axes[2].grid(True)

    axes[3].plot(t, history.get("dg_reward", []))
    axes[3].set_ylabel("€")
    axes[3].set_title("Distribution Grid Profit (Per Step)")
    axes[3].grid(True)

    axes[4].plot(t, history.get("dg_reward_cum", []))
    axes[4].set_ylabel("€")
    axes[4].set_title("Cumulative Distribution Grid Profit")
    axes[4].set_xlabel("Time Step")
    axes[4].grid(True)

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# Rollout Data Containers (Required by PPO)

@dataclass
class RolloutStep:
    """Stores data collected during a single environment step."""
    mg_obs: np.ndarray  # Shape: [N, mg_obs_dim]
    dg_obs: np.ndarray  # Shape: [dg_obs_dim]
    s_global: np.ndarray  # Shape: [global_dim]
    mg_action: np.ndarray  # Shape: [N, 2] (int64)
    dg_action: np.ndarray  # Shape: [2] (int64)
    mg_logp: np.ndarray  # Shape: [N] (float32) for PPO ratio calculation
    dg_logp: np.ndarray  # Shape: [1] (float32) for PPO ratio calculation
    reward_team: np.ndarray  # Shape: [1]
    done: np.ndarray  # Shape: [1]
    value: np.ndarray  # Shape: [1]


@dataclass
class RolloutBatch:
    """Stores batched rollout data collected over an entire episode/horizon."""
    mg_obs: np.ndarray
    dg_obs: np.ndarray
    s_global: np.ndarray
    mg_action: np.ndarray
    dg_action: np.ndarray
    mg_logp: np.ndarray
    dg_logp: np.ndarray
    reward_team: np.ndarray
    done: np.ndarray
    value: np.ndarray


def stack_rollouts(steps: List[RolloutStep]) -> RolloutBatch:
    """Converts a list of RolloutStep objects into a unified RolloutBatch tensor-like structure."""

    def _stack(name: str) -> np.ndarray: return np.stack([getattr(s, name) for s in steps], axis=0)

    return RolloutBatch(
        mg_obs=_stack("mg_obs").astype(np.float32),
        dg_obs=_stack("dg_obs").astype(np.float32),
        s_global=_stack("s_global").astype(np.float32),
        mg_action=_stack("mg_action").astype(np.int64),
        dg_action=_stack("dg_action").astype(np.int64),
        mg_logp=_stack("mg_logp").astype(np.float32),
        dg_logp=_stack("dg_logp").astype(np.float32),
        reward_team=_stack("reward_team").astype(np.float32),
        done=_stack("done").astype(np.float32),
        value=_stack("value").astype(np.float32),
    )

# Algorithm Configuration Configuration

@dataclass
class CTDEConfig:
    """Hyperparameters and settings for the MAPPO+CTDE architecture."""
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    horizon: int = 291
    updates: int = 1000

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    vf_coef: float = 0.5

    # MAPPO Hyperparameters
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    clip_param: float = 0.2

    w_mg: float = 1.0
    w_dg: float = 1.0

    # Nash Bargaining Configs
    use_nash: bool = True
    nash_ref_mg: float = 0.0
    nash_ref_dg: float = 0.0
    nash_scale_mg: float = 1.0
    nash_scale_dg: float = 1.0
    nash_eps: float = 1e-6

    # Output and Logging Options
    out_dir: str = "MAPPO_RBF_1000times"
    log_every: int = 20
    eval_every: int = 20
    save_models: bool = True

    # Classical Kernel (RBF) Regularization for Critic
    use_classical_kernel: bool = True
    kernel_reg_coef: float = 1e-3
    kernel_latent_dim: int = 4
    kernel_gamma: float = 1.0
    kernel_max_samples: int = 32
    kernel_eval_every: int = 1  # Evaluates at the start of each MAPPO update

# Main MAPPO Trainer (4 MG + 1 DG)

class CTDE_MAPPO_Trainer4MG:
    """
    Centralized Trainer implementing the MAPPO algorithm.
    Manages 4 Independent Microgrid Actors, 1 Distribution Grid Actor, and 1 Centralized Critic.
    """

    def __init__(self, mg_envs: List[Any], dg_env: Any, cfg: CTDEConfig):
        assert len(mg_envs) == 4, f"Expected exactly 4 microgrid environments, but got {len(mg_envs)}."
        self.mg_envs = mg_envs
        self.dg_env = dg_env
        self.cfg = cfg
        self.n_mg = 4
        os.makedirs(cfg.out_dir, exist_ok=True)

        self._seed_everything(cfg.seed)

        # Initialize environments to extract observation dimensions
        mg_obs0 = self.mg_envs[0].reset()
        dg_obs0 = self.dg_env.reset()

        self.mg_obs_dim = flatten_mg_obs(mg_obs0).shape[0]
        self.dg_obs_dim = ensure_1d(dg_obs0).shape[0]
        # Global state dimension includes all agents' observations + extra contextual parameters
        self.global_dim = self.n_mg * self.mg_obs_dim + self.dg_obs_dim + 3

        # Instantiate independent actor networks
        self.mg_actors: List[CategoricalActor] = [
            CategoricalActor(self.mg_obs_dim, nvec=(3, 3)).to(cfg.device) for _ in range(self.n_mg)
        ]
        self.dg_actor = CategoricalActor(self.dg_obs_dim, nvec=(3, 3)).to(cfg.device)

        # Initialize optimizers
        self.mg_opts: List[optim.Optimizer] = [
            optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.mg_actors
        ]
        self.dg_opt = optim.Adam(self.dg_actor.parameters(), lr=cfg.actor_lr)

        # Instantiate centralized critic network
        self.critic = LatentCentralCritic(
            global_dim=self.global_dim,
            latent_dim=cfg.kernel_latent_dim,
            hidden=(256, 256),
        ).to(cfg.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Initialize classical RBF kernel (if enabled)
        self.kernel = ClassicalKernelRegularizer(
            feature_dim=cfg.kernel_latent_dim,
            gamma=cfg.kernel_gamma,
        ) if cfg.use_classical_kernel else None

        self._update_idx = 0

        # Data logging dictionary structure
        self.log: TDict[str, List[float]] = {
            "episode_return_team": [], "mg_actor_loss": [], "dg_actor_loss": [],
            "entropy_mg": [], "entropy_dg": [],
            "critic_loss": [], "critic_td_loss": [], "critic_kernel_loss": [],
            "critic_grad_norm": [], "update_time_sec": [],
            "mg_actor_loss_mg1": [], "mg_actor_loss_mg2": [], "mg_actor_loss_mg3": [], "mg_actor_loss_mg4": [],
            "entropy_mg1": [], "entropy_mg2": [], "entropy_mg3": [], "entropy_mg4": [],
        }

        # Verify DG Environment API support
        self._dg_two_phase = hasattr(self.dg_env, "apply_price_action") and hasattr(self.dg_env, "settle")
        if not self._dg_two_phase:
            print("[Warning] DG environment does NOT implement 'apply_price_action/settle'. "
                  "The trainer will fall back to the standard 'step() + set_grid_power' 1-step lag execution.")

    def _seed_everything(self, seed: int):
        """Secures reproducibility by seeding Python, NumPy, and PyTorch RNGs."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _refresh_mg_obs(self, env: Any) -> Optional[dict]:
        """Attempts to safely fetch the latest observation dictionary from a Microgrid environment."""
        if hasattr(env, "get_obs"):
            try:
                return env.get_obs()
            except Exception:
                return None
        if hasattr(env, "_get_observation"):
            try:
                return env._get_observation()
            except Exception:
                return None
        return None

    def _extract_grid_power(self, mg_info: Any) -> float:
        """Parses the 'info' dict of an MG to extract the grid_power value reliably."""
        try:
            if isinstance(mg_info, dict) and "power_balance" in mg_info:
                pb = mg_info["power_balance"]
                if isinstance(pb, dict):
                    return safe_float(pb.get("grid_power", 0.0))
            elif isinstance(mg_info, (list, tuple)) and len(mg_info) > 0 and isinstance(mg_info[0], dict):
                _i = mg_info[0]
                if "power_balance" in _i and isinstance(_i["power_balance"], dict):
                    return safe_float(_i["power_balance"].get("grid_power", 0.0))
                return safe_float(_i.get("grid_power", 0.0))
        except Exception:
            pass
        return 0.0

    def _build_global(self, mg_obs_list: List[TDict[str, np.ndarray]], dg_obs: np.ndarray,
                      buy_price: float, sell_price: float, total_grid_power_prev: float) -> np.ndarray:
        """Concatenates localized observations to construct the global CTDE state representation."""
        mg_flats = [flatten_mg_obs(o) for o in mg_obs_list]
        dg_flat = ensure_1d(dg_obs)
        extra = np.array([buy_price, sell_price, total_grid_power_prev], dtype=np.float32)
        return np.concatenate([*mg_flats, dg_flat, extra], axis=0).astype(np.float32)

    def collect_rollout(self) -> Tuple[RolloutBatch, float, TDict[str, List[float]]]:
        """
        Executes a full episode (horizon) in the environment, collecting trajectories
        (observations, actions, rewards, log-probs) to train the PPO models.
        """
        cfg = self.cfg

        mg_obs_list = [env.reset() for env in self.mg_envs]
        dg_obs = self.dg_env.reset()

        if hasattr(self.dg_env, "set_grid_power"):
            self.dg_env.set_grid_power(0.0)

        prev_buy_price, prev_sell_price, prev_total_grid_power = 0.0, 0.0, 0.0

        history = {
            "buy_price": [], "sell_price": [], "grid_power": [],
            "upstream_power": [], "dg_reward": [], "dg_reward_cum": [],
        }
        dg_profit_cum = 0.0
        steps: List[RolloutStep] = []

        for _t in range(cfg.horizon):
            dg_obs_vec = ensure_1d(dg_obs)
            s_global_curr = self._build_global(
                mg_obs_list, dg_obs_vec, prev_buy_price, prev_sell_price, prev_total_grid_power
            )

            # Predict Value from Central Critic
            with torch.no_grad():
                v_curr = self.critic(torch.tensor(s_global_curr, device=cfg.device).unsqueeze(0)).item()

            # DG Agent Action (Record logp for PPO)
            dg_obs_t = torch.tensor(dg_obs_vec, device=cfg.device).unsqueeze(0)
            dg_action_t, dg_logp_t = self.dg_actor.act(dg_obs_t, deterministic=False)
            dg_action = dg_action_t.squeeze(0).cpu().numpy().astype(np.int64)
            dg_logp = dg_logp_t.squeeze(0).cpu().numpy()

            dg_reward, dg_done, dg_info, dg_obs_next = 0.0, False, {}, dg_obs

            # Execute DG action and propagate prices
            if self._dg_two_phase:
                try:
                    _ = self.dg_env.apply_price_action(dg_action)
                except Exception as e:
                    print(f"[Warning] dg_env.apply_price_action failed: {e}. Defaulting to step().")
                    dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.step(dg_action)
            else:
                dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.step(dg_action)

            buy_price = safe_float(getattr(self.dg_env, "buy_price", dg_info.get("buy_price", 0.0)))
            sell_price = safe_float(getattr(self.dg_env, "sell_price", dg_info.get("sell_price", 0.0)))

            # Broadcast updated prices to all Microgrids
            for i, env in enumerate(self.mg_envs):
                if hasattr(env, "set_grid_price"):
                    env.set_grid_price(buy_price, sell_price)
                refreshed = self._refresh_mg_obs(env)
                if isinstance(refreshed, dict):
                    mg_obs_list[i] = refreshed

            mg_flats = np.stack([flatten_mg_obs(o) for o in mg_obs_list], axis=0).astype(np.float32)

            # MGs Action Execution (Record logps for PPO)
            mg_actions = np.zeros((self.n_mg, 2), dtype=np.int64)
            mg_logps = np.zeros((self.n_mg,), dtype=np.float32)

            for i in range(self.n_mg):
                obs_i = torch.tensor(mg_flats[i], device=cfg.device).unsqueeze(0)
                a_i, logp_i = self.mg_actors[i].act(obs_i, deterministic=False)
                mg_actions[i] = a_i.squeeze(0).cpu().numpy().astype(np.int64)
                mg_logps[i] = logp_i.squeeze(0).cpu().numpy()

            mg_obs_next_list: List[TDict[str, np.ndarray]] = []
            mg_reward_sum = 0.0
            mg_done_any = False
            total_grid_power = 0.0

            # Step all Microgrids
            for i, env in enumerate(self.mg_envs):
                obs_next, r, d, info = env.step(mg_actions[i])
                mg_obs_next_list.append(obs_next)
                mg_reward_sum += float(r)
                mg_done_any = mg_done_any or bool(d)
                total_grid_power += float(self._extract_grid_power(info))

            # DG Settlement Phase
            if self._dg_two_phase:
                try:
                    dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.settle(total_grid_power)
                except Exception as e:
                    print(f"[Warning] dg_env.settle failed: {e}. Falling back to 1-step lag injection.")
                    if hasattr(self.dg_env, "set_grid_power"):
                        self.dg_env.set_grid_power(total_grid_power)
                    dg_obs_next = dg_obs
                    dg_reward = 0.0
                    dg_done = False
                    dg_info = {}
            else:
                if hasattr(self.dg_env, "set_grid_power"):
                    self.dg_env.set_grid_power(total_grid_power)

            # Define Team Objectives
            if getattr(cfg, "use_nash", False):
                reward_team = _nash_team_reward(float(mg_reward_sum), float(dg_reward), cfg)
            else:
                reward_team = cfg.w_mg * float(mg_reward_sum) + cfg.w_dg * float(dg_reward)

            done = bool(mg_done_any or dg_done)

            # Save step transition
            steps.append(
                RolloutStep(
                    mg_obs=mg_flats,
                    dg_obs=dg_obs_vec,
                    s_global=s_global_curr,
                    mg_action=mg_actions,
                    dg_action=dg_action,
                    mg_logp=mg_logps,
                    dg_logp=np.array([dg_logp], dtype=np.float32),
                    reward_team=np.array([reward_team], dtype=np.float32),
                    done=np.array([done], dtype=np.float32),
                    value=np.array([v_curr], dtype=np.float32),
                )
            )

            # Update histories and tracking variables
            dg_profit_cum += float(dg_reward)
            history["buy_price"].append(buy_price)
            history["sell_price"].append(sell_price)
            history["grid_power"].append(float(total_grid_power))

            up_p = safe_float(dg_info.get("upstream_power", 0.0)) if isinstance(dg_info, dict) else 0.0
            history["upstream_power"].append(up_p)
            history["dg_reward"].append(float(dg_reward))
            history["dg_reward_cum"].append(float(dg_profit_cum))

            # Move to next state
            mg_obs_list = mg_obs_next_list
            dg_obs = dg_obs_next
            prev_buy_price = buy_price
            prev_sell_price = sell_price
            prev_total_grid_power = float(total_grid_power)

            # Auto-reset upon episode completion
            if done:
                mg_obs_list = [env.reset() for env in self.mg_envs]
                dg_obs = self.dg_env.reset()
                if hasattr(self.dg_env, "set_grid_power"):
                    self.dg_env.set_grid_power(0.0)
                prev_buy_price = 0.0
                prev_sell_price = 0.0
                prev_total_grid_power = 0.0

        # Stack the entire buffer
        batch = stack_rollouts(steps)

        # Estimate Bootstrap Value for GAE calculation
        last_global = self._build_global(
            mg_obs_list, ensure_1d(dg_obs), prev_buy_price, prev_sell_price, prev_total_grid_power
        )
        with torch.no_grad():
            last_value = self.critic(torch.tensor(last_global, device=cfg.device).unsqueeze(0)).item()

        return batch, float(last_value), history

    def _compute_kernel_smoothness_loss(self, s_global_t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculates the RBF regularization loss to enforce smooth critic approximations.
        Sub-samples points to maintain computational efficiency.
        """
        cfg = self.cfg
        if (not cfg.use_classical_kernel) or self.kernel is None:
            return torch.zeros((), device=s_global_t.device, dtype=v_pred.dtype)
        if not self.kernel.enabled:
            return torch.zeros((), device=s_global_t.device, dtype=v_pred.dtype)

        n = s_global_t.shape[0]
        m = min(max(int(cfg.kernel_max_samples), 2), n)
        if m < 2:
            return torch.zeros((), device=s_global_t.device, dtype=v_pred.dtype)

        # Randomly sample 'm' indices
        if m < n:
            idx = torch.randperm(n, device=s_global_t.device)[:m]
        else:
            idx = torch.arange(n, device=s_global_t.device)

        z_sub = self.critic.encode(s_global_t[idx])
        v_sub = v_pred[idx]

        with torch.no_grad():
            # Build Kernel Similarity Matrix
            k = self.kernel.matrix(z_sub)
            k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

            # Mask out self-similarity (diagonal)
            eye = torch.eye(k.shape[0], device=k.device, dtype=k.dtype)
            k = k * (1.0 - eye)

            # Row normalization
            row_sum = k.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            w = k / row_sum

        # Calculate difference penalty weighted by the kernel matrix
        diff2 = (v_sub[:, None] - v_sub[None, :]).pow(2)
        return (w * diff2).sum(dim=-1).mean()

    def update(self, batch: RolloutBatch, last_value: float):
        """
        Main PPO optimization loop. Updates Actor (MGs & DG) and Critic networks
        using GAE targets and clipped surrogate losses.
        """
        cfg = self.cfg
        t0 = time.time()
        self._update_idx += 1

        rewards = batch.reward_team.squeeze(-1)
        dones = batch.done.squeeze(-1)
        values = batch.value.squeeze(-1)

        # Compute Advantages (GAE)
        adv, ret = compute_gae(rewards, dones, values, last_value, cfg.gamma, cfg.gae_lambda)
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Move to GPU/CPU tensors
        s_global_t = torch.tensor(batch.s_global, device=cfg.device)
        ret_t = torch.tensor(ret, device=cfg.device)
        adv_t = torch.tensor(adv, device=cfg.device)

        mg_obs_t = torch.tensor(batch.mg_obs, device=cfg.device)
        mg_action_t = torch.tensor(batch.mg_action, device=cfg.device)
        mg_logp_old = torch.tensor(batch.mg_logp, device=cfg.device)

        dg_obs_t = torch.tensor(batch.dg_obs, device=cfg.device)
        dg_action_t = torch.tensor(batch.dg_action, device=cfg.device)
        dg_logp_old = torch.tensor(batch.dg_logp, device=cfg.device).squeeze(-1)

        dataset_size = s_global_t.size(0)
        indices = np.arange(dataset_size)

        # Metric accumulators for tracing network performance across epochs
        epoch_c_loss, epoch_c_td_loss, epoch_c_kernel_loss = 0.0, 0.0, 0.0
        epoch_dg_loss = 0.0
        epoch_mg_losses = [0.0] * self.n_mg
        epoch_mg_ents = [0.0] * self.n_mg
        epoch_dg_ent = 0.0
        grad_norms = 0.0
        updates_count = 0

        # Calculate the RBF kernel penalty ONCE per PPO update to prevent
        # redundant graph calculations and speed up training.
        critic_kernel_loss_val = 0.0

        for epoch_idx in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                idx = indices[start:end]

                mb_s_global = s_global_t[idx]
                mb_ret = ret_t[idx]
                mb_adv = adv_t[idx]

                # Critic Update
                v_pred = self.critic(mb_s_global)
                critic_td_loss = 0.5 * (v_pred - mb_ret).pow(2).mean() * cfg.vf_coef

                # Compute kernel penalty strictly on the first mini-batch of the first epoch
                if epoch_idx == 0 and start == 0:
                    critic_kernel_loss_tensor = self._compute_kernel_smoothness_loss(
                        mb_s_global, v_pred
                    ) * cfg.kernel_reg_coef
                    critic_kernel_loss_val = critic_kernel_loss_tensor.item()
                else:
                    critic_kernel_loss_tensor = torch.tensor(0.0, device=cfg.device)

                critic_loss = critic_td_loss + critic_kernel_loss_tensor

                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                gn = float(nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm).item())
                self.critic_opt.step()

                grad_norms += gn
                epoch_c_loss += critic_loss.item()
                epoch_c_td_loss += critic_td_loss.item()
                epoch_c_kernel_loss += critic_kernel_loss_val

                # MG Actors Update (PPO Clipped Objective)
                for i in range(self.n_mg):
                    mb_mg_obs = mg_obs_t[idx, i, :]
                    mb_mg_act = mg_action_t[idx, i, :]
                    mb_mg_old_logp = mg_logp_old[idx, i]

                    new_logp = self.mg_actors[i].log_prob(mb_mg_obs, mb_mg_act)
                    entropy = self.mg_actors[i].entropy(mb_mg_obs).mean()

                    # Likelihood Ratio
                    ratio = torch.exp(new_logp - mb_mg_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * mb_adv

                    # Policy Loss
                    actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * entropy

                    self.mg_opts[i].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.mg_actors[i].parameters(), cfg.max_grad_norm)
                    self.mg_opts[i].step()

                    epoch_mg_losses[i] += actor_loss.item()
                    epoch_mg_ents[i] += entropy.item()

                # DG Actor Update (PPO Clipped Objective)
                mb_dg_obs = dg_obs_t[idx]
                mb_dg_act = dg_action_t[idx]
                mb_dg_old_logp = dg_logp_old[idx]

                new_dg_logp = self.dg_actor.log_prob(mb_dg_obs, mb_dg_act)
                dg_entropy = self.dg_actor.entropy(mb_dg_obs).mean()

                dg_ratio = torch.exp(new_dg_logp - mb_dg_old_logp)
                dg_surr1 = dg_ratio * mb_adv
                dg_surr2 = torch.clamp(dg_ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * mb_adv

                dg_actor_loss = -torch.min(dg_surr1, dg_surr2).mean() - cfg.ent_coef * dg_entropy

                self.dg_opt.zero_grad(set_to_none=True)
                dg_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.dg_actor.parameters(), cfg.max_grad_norm)
                self.dg_opt.step()

                epoch_dg_loss += dg_actor_loss.item()
                epoch_dg_ent += dg_entropy.item()
                updates_count += 1

        # Compute Metrics Logging Averages
        avg_c_loss = epoch_c_loss / updates_count
        avg_c_td_loss = epoch_c_td_loss / updates_count
        avg_c_kernel_loss = epoch_c_kernel_loss / updates_count
        avg_gn = grad_norms / updates_count
        avg_dg_loss = epoch_dg_loss / updates_count
        avg_dg_ent = epoch_dg_ent / updates_count

        avg_mg_losses = [l / updates_count for l in epoch_mg_losses]
        avg_mg_ents = [e / updates_count for e in epoch_mg_ents]

        # Log metrics to memory buffer
        self.log["critic_loss"].append(float(avg_c_loss))
        self.log["critic_td_loss"].append(float(avg_c_td_loss))
        self.log["critic_kernel_loss"].append(float(avg_c_kernel_loss))
        print(f"------ Critic Kernel Loss (Epoch Mean) = {avg_c_kernel_loss:.4f} ------")
        self.log["critic_grad_norm"].append(avg_gn)

        self.log["mg_actor_loss"].append(float(np.mean(avg_mg_losses)))
        self.log["entropy_mg"].append(float(np.mean(avg_mg_ents)))
        for i in range(self.n_mg):
            self.log[f"mg_actor_loss_mg{i + 1}"].append(avg_mg_losses[i])
            self.log[f"entropy_mg{i + 1}"].append(avg_mg_ents[i])

        self.log["dg_actor_loss"].append(float(avg_dg_loss))
        self.log["entropy_dg"].append(float(avg_dg_ent))
        self.log["update_time_sec"].append(float(time.time() - t0))

    def train(self):
        """Top-level function to trigger the MAPPO training loop over 'updates' iterations."""
        cfg = self.cfg
        for u in range(1, cfg.updates + 1):
            batch, last_value, history = self.collect_rollout()
            ep_ret = float(batch.reward_team.sum())
            self.log["episode_return_team"].append(ep_ret)

            self.update(batch, last_value)

            # Console progress reporting
            if (u % cfg.log_every) == 0:
                print(
                    f"[{u:04d}/{cfg.updates}] "
                    f"R_team={ep_ret:.3f} | "
                    f"vL={self.log['critic_loss'][-1]:.4f} | "
                    f"mgL={self.log['mg_actor_loss'][-1]:.4f} "
                    f"dgL={self.log['dg_actor_loss'][-1]:.4f} | "
                    f"ent(mg,dg)=({self.log['entropy_mg'][-1]:.3f},{self.log['entropy_dg'][-1]:.3f}) | "
                    f"upd={self.log['update_time_sec'][-1]:.3f}s"
                )

            # Regular Evaluation Checkpoints
            if cfg.eval_every > 0 and ((u % cfg.eval_every) == 0 or u == 1):
                try:
                    dg_fig_path = os.path.join(cfg.out_dir, f"dg_trace_u{u}.png")
                    plot_distribution_grid_results(history, save_path=dg_fig_path)

                    dg_csv_path = os.path.join(cfg.out_dir, f"dg_trace_u{u}.csv")
                    export_history_csv(history, dg_csv_path)
                except Exception as e:
                    print(f"[Warning] DG automated plotting failed: {e}")

                # Instruct individual Microgrid environments to export their own metrics
                for i, env in enumerate(self.mg_envs, start=1):
                    try:
                        if hasattr(env, "export_timeseries_csv"):
                            csv_path = os.path.join(cfg.out_dir, f"mg{i}_timeseries_u{u}.csv")
                            env.export_timeseries_csv(csv_path)
                        if hasattr(env, "plot_operation"):
                            fig = env.plot_operation()
                            if hasattr(fig, "write_html"):
                                fig.write_html(os.path.join(cfg.out_dir, f"mg{i}_timeseries_u{u}.html"))
                    except Exception as e:
                        print(f"[Warning] MG{i} custom 'plot_operation' failed: {e}")

        # Post-Training Procedures
        plot_training_curves(self.log, save_path=os.path.join(cfg.out_dir, "training_curves.png"))
        export_training_log_csv(self.log, os.path.join(cfg.out_dir, "training_log.csv"))

        if cfg.save_models:
            for i, actor in enumerate(self.mg_actors, start=1):
                torch.save(actor.state_dict(), os.path.join(cfg.out_dir, f"mg_actor_mg{i}.pt"))
            torch.save(self.dg_actor.state_dict(), os.path.join(cfg.out_dir, "dg_actor.pt"))
            torch.save(self.critic.state_dict(), os.path.join(cfg.out_dir, "central_critic.pt"))

        print(f"Training completed successfully. Saved output artifacts to: {cfg.out_dir}")

# Entry Point Execution
if __name__ == "__main__":
    # Ensure standard custom environment classes are imported
    from microgrid_environment8 import MicrogridEnv, DistributionGridEnv
    from config import MicrogridConfig

    data_path = "data/data_training/environment_table/Environment_data_2019.csv"

    # Compile simulation environment configuration
    microgrid_config = {
        "pv_capacity": MicrogridConfig.pv_capacity,
        "wind_capacity": MicrogridConfig.wind_capacity,
        "diesel_capacity": MicrogridConfig.diesel_capacity,
        "battery_capacity": MicrogridConfig.battery_capacity,
        "battery_soc_min": MicrogridConfig.battery_soc_min,
        "battery_soc_max": MicrogridConfig.battery_soc_max,
        "battery_efficiency": MicrogridConfig.battery_efficiency,
        "battery_max_power_ratio": MicrogridConfig.battery_max_power_ratio,
        "diesel_cost": MicrogridConfig.diesel_cost,
        "grid_buy_price": MicrogridConfig.grid_buy_price,
        "grid_sell_price": MicrogridConfig.grid_sell_price,
        "curtailment_penalty": MicrogridConfig.curtailment_penalty,
        "load_shedding_penalty": MicrogridConfig.load_shedding_penalty,
        "load_demand_scale": MicrogridConfig.load_demand_scale,
        "dynamic_curtailment_penalty": MicrogridConfig.dynamic_curtailment_penalty,
        "diesel_fuel_capacity": MicrogridConfig.diesel_fuel_capacity,
        "fuel_consumption_rate": MicrogridConfig.fuel_consumption_rate,
        "battery_soc_penalty_high": MicrogridConfig.battery_soc_penalty_high,
        "battery_soc_penalty_low": MicrogridConfig.battery_soc_penalty_low,
        "line_capacity_pv": MicrogridConfig.line_capacity_pv,
        "line_capacity_wind": MicrogridConfig.line_capacity_wind,
        "line_capacity_diesel": MicrogridConfig.line_capacity_diesel,
        "line_capacity_battery": MicrogridConfig.line_capacity_battery,
        "line_capacity_grid": MicrogridConfig.line_capacity_grid,
        "line_capacity_load": MicrogridConfig.line_capacity_load,
        "line_safety_margin": MicrogridConfig.line_safety_margin,
        "line_violation_penalty_coef": 15.0,
    }

    # Initialize 4 microgrid environments and 1 distribution grid market
    mg_envs = [MicrogridEnv(data_path=data_path, config=microgrid_config, Start=_) for _ in range(4)]
    dg_env = DistributionGridEnv(
        init_buy_price=MicrogridConfig.grid_buy_price,
        init_sell_price=MicrogridConfig.grid_sell_price,
    )

    # Initialize Hyperparameters
    cfg = CTDEConfig(
        seed=114514,
        horizon=291,
        updates=1000,
        eval_every=10,
        out_dir="MAPPO_RBF_Baseline114514",
        save_models=True,
    )

    # Trigger Execution pipeline
    start_time = time.time()
    trainer = CTDE_MAPPO_Trainer4MG(mg_envs, dg_env, cfg)
    trainer.train()
    end_time = time.time()

    print(f"Total MAPPO Training Time: {end_time - start_time:.2f} seconds")