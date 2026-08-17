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

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def moving_average(x, window=10):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def export_history_csv(history: dict, csv_path: str, add_time_index: bool = True):
    if not isinstance(history, dict) or len(history) == 0:
        raise ValueError("history is empty or not a dict")

    lengths = [len(v) for v in history.values() if isinstance(v, list)]
    if len(lengths) == 0:
        raise ValueError("history has no list fields")
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
    lengths = [len(v) for v in log.values() if isinstance(v, list)]
    T = min(lengths) if lengths else 0
    if T == 0:
        raise ValueError("log has no list data")

    df = pd.DataFrame({k: v[:T] for k, v in log.items() if isinstance(v, list)})
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df

# Nash bargaining style team reward

def _softplus(x: float) -> float:
    if x > 50.0:
        return x
    if x < -50.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _nash_team_reward(mg_r: float, dg_r: float, cfg) -> float:
    u_mg = _softplus((mg_r - cfg.nash_ref_mg) / max(cfg.nash_scale_mg, 1e-12))
    u_dg = _softplus((dg_r - cfg.nash_ref_dg) / max(cfg.nash_scale_dg, 1e-12))
    return math.log(u_mg + cfg.nash_eps) + math.log(u_dg + cfg.nash_eps)

# Helpers: obs flattening

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_mg_obs(mg_obs: TDict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(mg_obs, dict):
        raise TypeError(f"Expected dict obs for MicrogridEnv, got: {type(mg_obs)}")
    keys = sorted(mg_obs.keys())
    parts: List[np.ndarray] = []
    for k in keys:
        v = _to_numpy(mg_obs[k]).reshape(-1)
        parts.append(v.astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def ensure_1d(x: Any, dtype=np.float32) -> np.ndarray:
    return _to_numpy(x).reshape(-1).astype(dtype)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# Classical Actor Networks

class MLP(nn.Module):
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


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, nvec: Tuple[int, ...] = (3, 3), hidden=(256, 256)):
        super().__init__()
        self.nvec = tuple(int(n) for n in nvec)
        self.backbone = MLP(obs_dim, hidden=hidden, activation=nn.Tanh)
        self.heads = nn.ModuleList([nn.Linear(hidden[-1], n) for n in self.nvec])

    def _dist(self, obs: torch.Tensor) -> List[torch.distributions.Categorical]:
        h = self.backbone(obs)
        return [torch.distributions.Categorical(logits=head(h)) for head in self.heads]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        dists = self._dist(obs)
        logps = []
        for i, dist in enumerate(dists):
            logps.append(dist.log_prob(action[:, i]))
        return torch.stack(logps, dim=-1).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dists = self._dist(obs)
        ents = [d.entropy() for d in dists]
        return torch.stack(ents, dim=-1).sum(dim=-1)


# Fully Parameterized Quantum Circuit (PQC) Critic - QISKIT VERSION

def create_qiskit_qnn(n_qubits: int) -> EstimatorQNN:
    qc = QuantumCircuit(n_qubits)
    x = ParameterVector('x', n_qubits)
    theta = ParameterVector('θ', n_qubits * 2)

    # 1. Encoding Block
    for i in range(n_qubits):
        qc.rx(x[i], i)
        qc.ry(x[i], i)
        qc.rz(x[i], i)

    # 2. Entanglement Block - Fully Connected
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc.cx(i, j)

    # 3. Variational Block
    w_idx = 0
    for i in range(n_qubits):
        qc.ry(theta[w_idx], i)
        qc.rz(theta[w_idx + 1], i)
        w_idx += 2

    observables = [SparsePauliOp.from_sparse_list([("Z", [i], 1)], num_qubits=n_qubits) for i in range(n_qubits)]

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=x,
        weight_params=theta
    )
    return qnn


class PQCCentralCritic(nn.Module):
    def __init__(self, global_dim: int, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits

        self.state_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_qubits)
        )

        # Qiskit ML Connection
        qnn = create_qiskit_qnn(n_qubits)
        # Random Initialize
        initial_weights = np.random.uniform(0, 2 * np.pi, n_qubits * 2)
        self.qnn = TorchConnector(qnn, initial_weights=initial_weights)

        self.value_out = nn.Linear(n_qubits, 1)

    def forward(self, s_global: torch.Tensor) -> torch.Tensor:
        x_angles = torch.tanh(self.state_encoder(s_global)) * torch.pi

        exp_vals = self.qnn(x_angles)

        if exp_vals.dtype != torch.float32:
            exp_vals = exp_vals.to(torch.float32)

        value = self.value_out(exp_vals).squeeze(-1)
        return value

# GAE
def compute_gae(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, last_value: float,
                gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = float(last_value) if t == T - 1 else float(values[t + 1])
        delta = float(rewards[t]) + gamma * next_value * next_non_terminal - float(values[t])
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae
    ret = adv + values.astype(np.float32)
    return adv.astype(np.float32), ret.astype(np.float32)


# Plot helpers
def plot_training_curves(log: TDict[str, List[float]], save_path: str = "training_curves.png"):
    raw = np.array(log.get("episode_return_team", []), dtype=np.float32)
    smooth = moving_average(raw, window=10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    if len(raw) > 0:
        axes[0].plot(np.arange(len(raw)), raw, alpha=0.35, label="raw")

    if len(smooth) > 0:
        axes[0].plot(np.arange(len(smooth)) + 9, smooth, label="MA(10)")

    axes[0].set_title("Episode Team Return (per update horizon)")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(np.arange(len(log.get("mg_actor_loss", []))), log.get("mg_actor_loss", []), label="MG actor (avg)")
    for k in ("mg_actor_loss_mg1", "mg_actor_loss_mg2", "mg_actor_loss_mg3", "mg_actor_loss_mg4"):
        if k in log and len(log[k]) > 0:
            axes[1].plot(np.arange(len(log[k])), log[k], label=k)
    axes[1].plot(np.arange(len(log.get("dg_actor_loss", []))), log.get("dg_actor_loss", []), label="DG actor")
    axes[1].plot(np.arange(len(log.get("critic_loss", []))), log.get("critic_loss", []), label="PQC Critic (TD Loss)")

    axes[1].set_title("Training Signals (per update)")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(np.arange(len(log.get("entropy_mg", []))), log.get("entropy_mg", []), label="MG entropy (avg)")
    for k in ("entropy_mg1", "entropy_mg2", "entropy_mg3", "entropy_mg4"):
        if k in log and len(log[k]) > 0:
            axes[2].plot(np.arange(len(log[k])), log[k], label=k)
    axes[2].plot(np.arange(len(log.get("entropy_dg", []))), log.get("entropy_dg", []), label="DG entropy")
    axes[2].plot(np.arange(len(log.get("update_time_sec", []))), log.get("update_time_sec", []), label="update_time(s)")
    axes[2].set_title("Entropy / Update Time (per update)")
    axes[2].set_xlabel("Update")
    axes[2].set_ylabel("Value")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_grid_results(history: TDict[str, List[float]], save_path: str | None = None, show: bool = False):
    t = np.arange(len(history.get("dg_reward", [])))
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(t, history.get("buy_price", []), label="Buy Price")
    axes[0].plot(t, history.get("sell_price", []), label="Sell Price")
    axes[0].set_ylabel("€/kWh")
    axes[0].set_title("Distribution Grid Pricing")
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
    axes[3].set_title("Distribution Grid Profit per Step")
    axes[3].grid(True)

    axes[4].plot(t, history.get("dg_reward_cum", []))
    axes[4].set_ylabel("€")
    axes[4].set_title("Cumulative Distribution Grid Profit")
    axes[4].set_xlabel("Time Step")
    axes[4].grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# Rollout containers

@dataclass
class RolloutStep:
    mg_obs: np.ndarray  # [N, mg_obs_dim]
    dg_obs: np.ndarray  # [dg_obs_dim]
    s_global: np.ndarray  # [global_dim]
    mg_action: np.ndarray  # [N, 2] int64
    dg_action: np.ndarray  # [2] int64
    reward_team: np.ndarray  # [1]
    done: np.ndarray  # [1]
    value: np.ndarray  # [1]


@dataclass
class RolloutBatch:
    mg_obs: np.ndarray
    dg_obs: np.ndarray
    s_global: np.ndarray
    mg_action: np.ndarray
    dg_action: np.ndarray
    reward_team: np.ndarray
    done: np.ndarray
    value: np.ndarray


def stack_rollouts(steps: List[RolloutStep]) -> RolloutBatch:
    def _stack(name: str) -> np.ndarray:
        return np.stack([getattr(s, name) for s in steps], axis=0)

    return RolloutBatch(
        mg_obs=_stack("mg_obs").astype(np.float32),
        dg_obs=_stack("dg_obs").astype(np.float32),
        s_global=_stack("s_global").astype(np.float32),
        mg_action=_stack("mg_action").astype(np.int64),
        dg_action=_stack("dg_action").astype(np.int64),
        reward_team=_stack("reward_team").astype(np.float32),
        done=_stack("done").astype(np.float32),
        value=_stack("value").astype(np.float32),
    )

# Config

@dataclass
class CTDEConfig:
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

    w_mg: float = 1.0
    w_dg: float = 1.0

    use_nash: bool = True
    nash_ref_mg: float = 0.0
    nash_ref_dg: float = 0.0
    nash_scale_mg: float = 1.0
    nash_scale_dg: float = 1.0
    nash_eps: float = 1e-6

    out_dir: str = "A2C_1000times_4"
    log_every: int = 1
    eval_every: int = 20
    save_models: bool = True

    # PQC specific configuration
    n_qubits: int = 4

# Trainer
class CTDE_A2C_Trainer4MG:
    def __init__(self, mg_envs: List[Any], dg_env: Any, cfg: CTDEConfig):
        assert len(mg_envs) == 4, f"Expected 4 microgrid envs, got {len(mg_envs)}"
        self.mg_envs = mg_envs
        self.dg_env = dg_env
        self.cfg = cfg
        self.n_mg = 4
        os.makedirs(cfg.out_dir, exist_ok=True)

        self._seed_everything(cfg.seed)

        mg_obs0 = self.mg_envs[0].reset()
        dg_obs0 = self.dg_env.reset()
        self.mg_obs_dim = flatten_mg_obs(mg_obs0).shape[0]
        self.dg_obs_dim = ensure_1d(dg_obs0).shape[0]
        self.global_dim = self.n_mg * self.mg_obs_dim + self.dg_obs_dim + 3

        self.mg_actors: List[CategoricalActor] = [
            CategoricalActor(self.mg_obs_dim, nvec=(3, 3)).to(cfg.device) for _ in range(self.n_mg)
        ]
        self.dg_actor = CategoricalActor(self.dg_obs_dim, nvec=(3, 3)).to(cfg.device)

        self.mg_opts: List[optim.Optimizer] = [
            optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.mg_actors
        ]
        self.dg_opt = optim.Adam(self.dg_actor.parameters(), lr=cfg.actor_lr)

        # Import the critic network using Qiskit PQC.
        self.critic = PQCCentralCritic(
            global_dim=self.global_dim,
            n_qubits=cfg.n_qubits
        ).to(cfg.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self._update_idx = 0

        self.log: TDict[str, List[float]] = {
            "episode_return_team": [],
            "mg_actor_loss": [],
            "dg_actor_loss": [],
            "entropy_mg": [],
            "entropy_dg": [],
            "critic_loss": [],
            "critic_grad_norm": [],
            "update_time_sec": [],
            "mg_actor_loss_mg1": [], "mg_actor_loss_mg2": [], "mg_actor_loss_mg3": [], "mg_actor_loss_mg4": [],
            "entropy_mg1": [], "entropy_mg2": [], "entropy_mg3": [], "entropy_mg4": [],
        }

        self._dg_two_phase = hasattr(self.dg_env, "apply_price_action") and hasattr(self.dg_env, "settle")
        if not self._dg_two_phase:
            print("[warn] DG env does NOT have apply_price_action/settle. Falling back to step().")

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _refresh_mg_obs(self, env: Any) -> Optional[dict]:
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
        mg_flats = [flatten_mg_obs(o) for o in mg_obs_list]
        dg_flat = ensure_1d(dg_obs)
        extra = np.array([buy_price, sell_price, total_grid_power_prev], dtype=np.float32)
        return np.concatenate([*mg_flats, dg_flat, extra], axis=0).astype(np.float32)

    def collect_rollout(self) -> Tuple[RolloutBatch, float, TDict[str, List[float]]]:
        cfg = self.cfg
        mg_obs_list = [env.reset() for env in self.mg_envs]
        dg_obs = self.dg_env.reset()

        if hasattr(self.dg_env, "set_grid_power"):
            self.dg_env.set_grid_power(0.0)

        prev_buy_price = 0.0
        prev_sell_price = 0.0
        prev_total_grid_power = 0.0

        history = {
            "buy_price": [], "sell_price": [], "grid_power": [],
            "upstream_power": [], "dg_reward": [], "dg_reward_cum": [],
        }
        dg_profit_cum = 0.0
        steps: List[RolloutStep] = []

        for _t in range(cfg.horizon):
            dg_obs_vec = ensure_1d(dg_obs)
            s_global_curr = self._build_global(
                mg_obs_list, dg_obs_vec, prev_buy_price, prev_sell_price, prev_total_grid_power,
            )

            with torch.no_grad():
                v_curr = self.critic(
                    torch.tensor(s_global_curr, device=cfg.device).unsqueeze(0)
                ).item()

            dg_obs_t = torch.tensor(dg_obs_vec, device=cfg.device).unsqueeze(0)
            dg_action_t, _ = self.dg_actor.act(dg_obs_t, deterministic=False)
            dg_action = dg_action_t.squeeze(0).cpu().numpy().astype(np.int64)

            dg_reward = 0.0
            dg_done = False
            dg_info: dict = {}
            dg_obs_next = dg_obs

            if self._dg_two_phase:
                try:
                    _ = self.dg_env.apply_price_action(dg_action)
                except Exception as e:
                    dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.step(dg_action)
            else:
                dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.step(dg_action)

            buy_price = safe_float(getattr(self.dg_env, "buy_price", dg_info.get("buy_price", 0.0)))
            sell_price = safe_float(getattr(self.dg_env, "sell_price", dg_info.get("sell_price", 0.0)))

            for i, env in enumerate(self.mg_envs):
                if hasattr(env, "set_grid_price"):
                    env.set_grid_price(buy_price, sell_price)
                refreshed = self._refresh_mg_obs(env)
                if isinstance(refreshed, dict):
                    mg_obs_list[i] = refreshed

            mg_flats = np.stack([flatten_mg_obs(o) for o in mg_obs_list], axis=0).astype(np.float32)
            mg_actions = np.zeros((self.n_mg, 2), dtype=np.int64)

            for i in range(self.n_mg):
                obs_i = torch.tensor(mg_flats[i], device=cfg.device).unsqueeze(0)
                a_i, _ = self.mg_actors[i].act(obs_i, deterministic=False)
                mg_actions[i] = a_i.squeeze(0).cpu().numpy().astype(np.int64)

            mg_obs_next_list: List[TDict[str, np.ndarray]] = []
            mg_reward_sum = 0.0
            mg_done_any = False
            total_grid_power = 0.0

            for i, env in enumerate(self.mg_envs):
                obs_next, r, d, info = env.step(mg_actions[i])
                mg_obs_next_list.append(obs_next)
                mg_reward_sum += float(r)
                mg_done_any = mg_done_any or bool(d)
                total_grid_power += float(self._extract_grid_power(info))

            if self._dg_two_phase:
                try:
                    dg_obs_next, dg_reward, dg_done, dg_info = self.dg_env.settle(total_grid_power)
                except Exception as e:
                    if hasattr(self.dg_env, "set_grid_power"):
                        self.dg_env.set_grid_power(total_grid_power)
                    dg_obs_next, dg_reward, dg_done = dg_obs, 0.0, False
            else:
                if hasattr(self.dg_env, "set_grid_power"):
                    self.dg_env.set_grid_power(total_grid_power)

            if getattr(cfg, "use_nash", False):
                reward_team = _nash_team_reward(float(mg_reward_sum), float(dg_reward), cfg)
            else:
                reward_team = cfg.w_mg * float(mg_reward_sum) + cfg.w_dg * float(dg_reward)

            done = bool(mg_done_any or dg_done)

            steps.append(
                RolloutStep(
                    mg_obs=mg_flats, dg_obs=dg_obs_vec, s_global=s_global_curr,
                    mg_action=mg_actions, dg_action=dg_action,
                    reward_team=np.array([reward_team], dtype=np.float32),
                    done=np.array([done], dtype=np.float32), value=np.array([v_curr], dtype=np.float32),
                )
            )

            dg_profit_cum += float(dg_reward)
            history["buy_price"].append(buy_price)
            history["sell_price"].append(sell_price)
            history["grid_power"].append(float(total_grid_power))
            history["upstream_power"].append(
                safe_float(dg_info.get("upstream_power", 0.0)) if isinstance(dg_info, dict) else 0.0)
            history["dg_reward"].append(float(dg_reward))
            history["dg_reward_cum"].append(float(dg_profit_cum))

            mg_obs_list, dg_obs = mg_obs_next_list, dg_obs_next
            prev_buy_price, prev_sell_price, prev_total_grid_power = buy_price, sell_price, float(total_grid_power)

            if done:
                mg_obs_list = [env.reset() for env in self.mg_envs]
                dg_obs = self.dg_env.reset()
                if hasattr(self.dg_env, "set_grid_power"):
                    self.dg_env.set_grid_power(0.0)
                prev_buy_price = prev_sell_price = prev_total_grid_power = 0.0

        batch = stack_rollouts(steps)

        last_global = self._build_global(
            mg_obs_list, ensure_1d(dg_obs), prev_buy_price, prev_sell_price, prev_total_grid_power,
        )
        with torch.no_grad():
            last_value = self.critic(
                torch.tensor(last_global, device=cfg.device).unsqueeze(0)
            ).item()

        return batch, float(last_value), history

    def update(self, batch: RolloutBatch, last_value: float):
        cfg = self.cfg
        t0 = time.time()
        self._update_idx += 1

        rewards = batch.reward_team.squeeze(-1)
        dones = batch.done.squeeze(-1)
        values = batch.value.squeeze(-1)

        adv, ret = compute_gae(rewards, dones, values, last_value, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s_global_t = torch.tensor(batch.s_global, device=cfg.device)
        ret_t = torch.tensor(ret, device=cfg.device)
        adv_t = torch.tensor(adv, device=cfg.device)

        v_pred = self.critic(s_global_t)

        # PQC's TD Loss
        critic_td_loss = 0.5 * (v_pred - ret_t).pow(2).mean() * cfg.vf_coef
        critic_loss = critic_td_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        gn = float(nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm).item())
        self.critic_opt.step()

        mg_obs_t = torch.tensor(batch.mg_obs, device=cfg.device)
        mg_action_t = torch.tensor(batch.mg_action, device=cfg.device)
        dg_obs_t = torch.tensor(batch.dg_obs, device=cfg.device)
        dg_action_t = torch.tensor(batch.dg_action, device=cfg.device)

        mg_losses, mg_ents = [], []
        for i in range(self.n_mg):
            logp_i = self.mg_actors[i].log_prob(mg_obs_t[:, i, :], mg_action_t[:, i, :])
            ent_i = self.mg_actors[i].entropy(mg_obs_t[:, i, :]).mean()
            loss_i = -(logp_i * adv_t).mean() - cfg.ent_coef * ent_i

            self.mg_opts[i].zero_grad(set_to_none=True)
            loss_i.backward()
            nn.utils.clip_grad_norm_(self.mg_actors[i].parameters(), cfg.max_grad_norm)
            self.mg_opts[i].step()

            mg_losses.append(float(loss_i.item()))
            mg_ents.append(float(ent_i.item()))

        dg_logp = self.dg_actor.log_prob(dg_obs_t, dg_action_t)
        dg_entropy = self.dg_actor.entropy(dg_obs_t).mean()
        dg_actor_loss = -(dg_logp * adv_t).mean() - cfg.ent_coef * dg_entropy

        self.dg_opt.zero_grad(set_to_none=True)
        dg_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.dg_actor.parameters(), cfg.max_grad_norm)
        self.dg_opt.step()

        self.log["critic_loss"].append(float(critic_loss.item()))
        self.log["critic_grad_norm"].append(gn)
        self.log["mg_actor_loss"].append(float(np.mean(mg_losses)))
        self.log["entropy_mg"].append(float(np.mean(mg_ents)))
        for i in range(self.n_mg):
            self.log[f"mg_actor_loss_mg{i + 1}"].append(mg_losses[i])
            self.log[f"entropy_mg{i + 1}"].append(mg_ents[i])

        self.log["dg_actor_loss"].append(float(dg_actor_loss.item()))
        self.log["entropy_dg"].append(float(dg_entropy.item()))
        self.log["update_time_sec"].append(float(time.time() - t0))

    def train(self):
        cfg = self.cfg
        for u in range(1, cfg.updates + 1):
            batch, last_value, history = self.collect_rollout()
            ep_ret = float(batch.reward_team.sum())
            self.log["episode_return_team"].append(ep_ret)

            self.update(batch, last_value)

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

            if cfg.eval_every > 0 and ((u % cfg.eval_every) == 0 or u == 1):
                try:
                    dg_fig_path = os.path.join(cfg.out_dir, f"dg_trace_u{u}.png")
                    plot_distribution_grid_results(history, save_path=dg_fig_path)

                    dg_csv_path = os.path.join(cfg.out_dir, f"dg_trace_u{u + 1}.csv")
                    export_history_csv(history, dg_csv_path)
                except Exception as e:
                    print(f"[warn] DG plotting failed: {e}")

                for i, env in enumerate(self.mg_envs, start=1):
                    try:
                        if hasattr(env, "export_timeseries_csv"):
                            csv_path = os.path.join(cfg.out_dir, f"mg{i}_timeseries_u{u + 1}.csv")
                            env.export_timeseries_csv(csv_path)
                        if hasattr(env, "plot_operation"):
                            fig = env.plot_operation()
                            if hasattr(fig, "write_html"):
                                fig.write_html(os.path.join(cfg.out_dir, f"mg{i}_timeseries_u{u}.html"))
                    except Exception as e:
                        print(f"[warn] MG{i} plot_operation failed: {e}")

        plot_training_curves(self.log, save_path=os.path.join(cfg.out_dir, "training_curves.png"))
        export_training_log_csv(self.log, os.path.join(cfg.out_dir, "training_log.csv"))

        if cfg.save_models:
            for i, actor in enumerate(self.mg_actors, start=1):
                torch.save(actor.state_dict(), os.path.join(cfg.out_dir, f"mg_actor_mg{i}.pt"))
            torch.save(self.dg_actor.state_dict(), os.path.join(cfg.out_dir, "dg_actor.pt"))
            torch.save(self.critic.state_dict(), os.path.join(cfg.out_dir, "central_critic.pt"))

        print(f"Saved outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    from microgrid_environment import MicrogridEnv, DistributionGridEnv
    from config import MicrogridConfig

    data_path = "data/data_training/environment_table/Environment_data_2019.csv"

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

    mg_envs = [MicrogridEnv(data_path=data_path, config=microgrid_config, Start=_) for _ in range(4)]
    dg_env = DistributionGridEnv(
        init_buy_price=MicrogridConfig.grid_buy_price,
        init_sell_price=MicrogridConfig.grid_sell_price,
    )

    cfg = CTDEConfig(
        seed=0,
        horizon=291,
        updates=1000,
        eval_every=10,
        out_dir="A2C_PQC",
        save_models=True,
        n_qubits=4
    )

    trainer = CTDE_A2C_Trainer4MG(mg_envs, dg_env, cfg)
    trainer.train()