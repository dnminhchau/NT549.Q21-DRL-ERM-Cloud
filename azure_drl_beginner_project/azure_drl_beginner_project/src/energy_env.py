from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    max_hosts: int = 8
    min_active_hosts: int = 1
    episode_length: int = 288
    dvfs_levels: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
    host_nominal_capacity: float = 1.0 / 8.0
    p_idle: float = 80.0
    p_peak: float = 200.0
    p_sleep: float = 10.0
    switch_cost: float = 8.0
    reward_w_energy: float = 1.0
    reward_w_sla: float = 4.0
    reward_w_switch: float = 0.15
    reward_w_util: float = 0.25
    seed: int = 42


class CloudEnergyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, workload: np.ndarray, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.workload = np.asarray(workload, dtype=np.float32).reshape(-1)
        if self.workload.size < self.config.episode_length + 1:
            raise ValueError(
                f"Workload quá ngắn: cần ít nhất {self.config.episode_length + 1} điểm, "
                f"hiện có {self.workload.size}."
            )

        self.rng = np.random.default_rng(self.config.seed)

        # 0: keep, 1: power on one host, 2: sleep one host, 3: dvfs up, 4: dvfs down
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([0.0] * 9, dtype=np.float32),
            high=np.array([2.0] * 9, dtype=np.float32),
            dtype=np.float32,
        )

        self.start_idx = 0
        self.step_idx = 0
        self.active_hosts = max(2, self.config.max_hosts // 2)
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self.last_power = 0.0
        self.last_sla = 0.0
        self.last_switches = 0
        self.trace: list[dict[str, Any]] = []

    @property
    def dvfs(self) -> float:
        return self.config.dvfs_levels[self.dvfs_idx]

    def _get_demand(self, offset: int = 0) -> float:
        idx = min(self.start_idx + self.step_idx + offset, self.workload.size - 1)
        return float(self.workload[idx])

    def _cluster_capacity(self) -> float:
        return self.active_hosts * self.config.host_nominal_capacity * self.dvfs

    def _compute_power(self, demand: float) -> tuple[float, float, np.ndarray]:
        active_hosts = max(self.active_hosts, 1)
        cluster_capacity = max(self._cluster_capacity(), 1e-8)
        offered_load = demand / cluster_capacity
        per_host_util = np.clip(offered_load, 0.0, 1.5)

        active_power = (
            self.config.p_idle
            + (self.config.p_peak - self.config.p_idle) * min(per_host_util, 1.0) * (self.dvfs ** 2)
        )
        sleeping_hosts = self.config.max_hosts - self.active_hosts
        total_power = self.active_hosts * active_power + sleeping_hosts * self.config.p_sleep

        host_utils = np.zeros(self.config.max_hosts, dtype=np.float32)
        if self.active_hosts > 0:
            host_utils[: self.active_hosts] = per_host_util

        return float(total_power), float(per_host_util), host_utils

    def _sla_violation(self, demand: float) -> float:
        cap = self._cluster_capacity()
        if cap <= 0:
            return 1.0
        return float(max(0.0, demand - cap) / max(demand, 1e-8))

    def _observation(self) -> np.ndarray:
        demand_now = self._get_demand(0)
        demand_next = self._get_demand(1)
        _, mean_util, host_utils = self._compute_power(demand_now)
        sleep_ratio = (self.config.max_hosts - self.active_hosts) / self.config.max_hosts
        obs = np.array(
            [
                demand_now,
                demand_next,
                self.active_hosts / self.config.max_hosts,
                self.dvfs / max(self.config.dvfs_levels),
                self.last_power / (self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2),
                self.last_sla,
                float(np.mean(host_utils[: self.active_hosts])) if self.active_hosts > 0 else 0.0,
                float(np.std(host_utils[: self.active_hosts])) if self.active_hosts > 0 else 0.0,
                sleep_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = self.workload.size - self.config.episode_length - 1
        self.start_idx = int(self.rng.integers(0, max(1, max_start)))
        self.step_idx = 0
        self.active_hosts = max(2, self.config.max_hosts // 2)
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self.last_power, _, _ = self._compute_power(self._get_demand(0))
        self.last_sla = self._sla_violation(self._get_demand(0))
        self.last_switches = 0
        self.trace = []
        return self._observation(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"

        switches = 0
        if action == 1 and self.active_hosts < self.config.max_hosts:
            self.active_hosts += 1
            switches = 1
        elif action == 2 and self.active_hosts > self.config.min_active_hosts:
            self.active_hosts -= 1
            switches = 1
        elif action == 3 and self.dvfs_idx < len(self.config.dvfs_levels) - 1:
            self.dvfs_idx += 1
        elif action == 4 and self.dvfs_idx > 0:
            self.dvfs_idx -= 1

        demand = self._get_demand(0)
        power, mean_util, host_utils = self._compute_power(demand)
        sla = self._sla_violation(demand)
        normalized_energy = power / (self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2)
        utilization_bonus = min(mean_util, 1.0)

        reward = -(
            self.config.reward_w_energy * normalized_energy
            + self.config.reward_w_sla * sla
            + self.config.reward_w_switch * switches
        ) + self.config.reward_w_util * utilization_bonus

        info = {
            "demand": demand,
            "power": power,
            "sla_violation": sla,
            "active_hosts": self.active_hosts,
            "dvfs": self.dvfs,
            "mean_util": mean_util,
            "switches": switches,
            "sla": sla,
        }
        self.trace.append(info)

        self.last_power = power
        self.last_sla = sla
        self.last_switches = switches
        self.step_idx += 1

        terminated = self.step_idx >= self.config.episode_length
        truncated = False
        obs = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.step_idx} active_hosts={self.active_hosts} dvfs={self.dvfs:.2f} "
            f"power={self.last_power:.2f} sla={self.last_sla:.4f}"
        )


def load_workload_csv(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    if "demand" not in df.columns:
        raise ValueError("CSV phải có cột 'demand'.")
    values = df["demand"].astype(float).to_numpy()
    return np.clip(values, 0.0, 2.0)
