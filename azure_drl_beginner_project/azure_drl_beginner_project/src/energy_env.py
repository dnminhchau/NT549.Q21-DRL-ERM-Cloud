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
    p_off: float = 1.5
    switch_cost: float = 8.0
    reward_w_energy: float = 1.0
    reward_w_sla: float = 4.0
    reward_w_switch: float = 0.15
    reward_w_util: float = 0.25
    reward_w_migration: float = 0.08
    cooling_factor: float = 0.22
    ups_overhead: float = 0.08
    base_ambient_temp_c: float = 24.0
    thermal_cap: float = 0.10
    degradation_temp_ref_c: float = 60.0
    degradation_q10: float = 2.0
    vm_granularity: float = 0.02
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

        # 0: keep, 1: power on one host, 2: sleep one host, 3: dvfs up, 4: dvfs down, 5: off one sleeping host
        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(
            low=np.array([0.0] * 12, dtype=np.float32),
            high=np.array([2.0] * 12, dtype=np.float32),
            dtype=np.float32,
        )

        self.start_idx = 0
        self.step_idx = 0
        self.active_hosts = max(2, self.config.max_hosts // 2)
        self.sleeping_hosts = self.config.max_hosts - self.active_hosts
        self.off_hosts = 0
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self.last_power = 0.0
        self.last_facility_power = 0.0
        self.last_pue = 1.0
        self.last_sla = 0.0
        self.last_switches = 0
        self.last_migrations = 0
        self.host_temperatures = np.full(self.config.max_hosts, self.config.base_ambient_temp_c, dtype=np.float32)
        self.last_hardware_wear = 0.0
        self.trace: list[dict[str, Any]] = []

    @property
    def dvfs(self) -> float:
        return self.config.dvfs_levels[self.dvfs_idx]

    def _get_demand(self, offset: int = 0) -> float:
        idx = min(self.start_idx + self.step_idx + offset, self.workload.size - 1)
        return float(self.workload[idx])

    def _cluster_capacity(self) -> float:
        return self.active_hosts * self.config.host_nominal_capacity * self.dvfs

    def _extract_vms(self, demand: float) -> np.ndarray:
        vm_size = max(self.config.vm_granularity, 1e-4)
        vm_count = max(1, int(np.ceil(demand / vm_size)))
        remaining = demand
        vms = []
        for _ in range(vm_count - 1):
            alloc = min(vm_size, remaining)
            vms.append(alloc)
            remaining -= alloc
        vms.append(max(remaining, 0.0))
        return np.array(vms, dtype=np.float32)

    def _allocate_vms(self, vms: np.ndarray, host_capacity: float) -> tuple[np.ndarray, int]:
        host_loads = np.zeros(self.config.max_hosts, dtype=np.float32)
        if self.active_hosts <= 0:
            return host_loads, int(vms.size)

        active_slice = slice(0, self.active_hosts)
        migrations = 0
        for vm in np.sort(vms)[::-1]:
            residual = host_capacity - host_loads[active_slice]
            fit_idx = np.where(residual >= vm)[0]
            if fit_idx.size > 0:
                best_local = int(fit_idx[np.argmin(residual[fit_idx] - vm)])
                host_loads[best_local] += vm
            else:
                # Spillover to least loaded host = overloaded placement + migration pressure
                best_local = int(np.argmin(host_loads[active_slice]))
                host_loads[best_local] += vm
                migrations += 1

        return host_loads, migrations

    def _compute_power(self, demand: float) -> tuple[float, float, np.ndarray, int]:
        if self.active_hosts <= 0:
            sleeping_hosts = self.sleeping_hosts
            total_power = sleeping_hosts * self.config.p_sleep + self.off_hosts * self.config.p_off
            return float(total_power), 0.0, np.zeros(self.config.max_hosts, dtype=np.float32), 0

        host_capacity = self.config.host_nominal_capacity * self.dvfs
        vm_list = self._extract_vms(demand)
        host_loads, migrations = self._allocate_vms(vm_list, host_capacity)

        active_utils = np.clip(host_loads[: self.active_hosts] / max(host_capacity, 1e-8), 0.0, 1.8)
        active_power_each = self.config.p_idle + (self.config.p_peak - self.config.p_idle) * np.minimum(active_utils, 1.0) * (
            self.dvfs ** 2
        )

        total_power = float(np.sum(active_power_each))
        total_power += self.sleeping_hosts * self.config.p_sleep + self.off_hosts * self.config.p_off

        host_utils = np.zeros(self.config.max_hosts, dtype=np.float32)
        host_utils[: self.active_hosts] = active_utils
        mean_util = float(np.mean(active_utils)) if self.active_hosts > 0 else 0.0

        return float(total_power), mean_util, host_utils, int(migrations)

    def _facility_power(self, it_power: float, mean_util: float) -> tuple[float, float]:
        cooling_multiplier = 1.0 + self.config.cooling_factor * max(mean_util, 0.0)
        infra_multiplier = 1.0 + self.config.ups_overhead
        facility = it_power * cooling_multiplier * infra_multiplier
        pue = facility / max(it_power, 1e-8)
        return float(facility), float(pue)

    def _update_thermal_state(self, host_utils: np.ndarray):
        ambient = self.config.base_ambient_temp_c
        target_temp = ambient + 42.0 * np.clip(host_utils, 0.0, 1.5) * (self.dvfs ** 2)
        self.host_temperatures = self.host_temperatures + self.config.thermal_cap * (target_temp - self.host_temperatures)

    def _hardware_wear(self) -> float:
        # Q10 rule: degradation speed doubles every +10C compared to reference
        over_ref = (self.host_temperatures - self.config.degradation_temp_ref_c) / 10.0
        accel = np.power(self.config.degradation_q10, np.clip(over_ref, -5.0, 6.0))
        return float(np.mean(accel))

    def _sla_violation(self, demand: float) -> float:
        cap = self._cluster_capacity()
        if cap <= 0:
            return 1.0
        return float(max(0.0, demand - cap) / max(demand, 1e-8))

    def _observation(self) -> np.ndarray:
        demand_now = self._get_demand(0)
        demand_next = self._get_demand(1)
        _, mean_util, host_utils, _ = self._compute_power(demand_now)
        sleep_ratio = self.sleeping_hosts / self.config.max_hosts
        off_ratio = self.off_hosts / self.config.max_hosts
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
                off_ratio,
                self.last_pue / 2.5,
                float(np.mean(self.host_temperatures)) / 100.0,
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
        self.sleeping_hosts = self.config.max_hosts - self.active_hosts
        self.off_hosts = 0
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self.last_power, mean_util, host_utils, migrations = self._compute_power(self._get_demand(0))
        self.last_facility_power, self.last_pue = self._facility_power(self.last_power, mean_util)
        self._update_thermal_state(host_utils)
        self.last_hardware_wear = self._hardware_wear()
        self.last_migrations = migrations
        self.last_sla = self._sla_violation(self._get_demand(0))
        self.last_switches = 0
        self.trace = []
        return self._observation(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"

        switches = 0
        if action == 1:
            if self.sleeping_hosts > 0:
                self.sleeping_hosts -= 1
                self.active_hosts += 1
                switches = 1
            elif self.off_hosts > 0:
                self.off_hosts -= 1
                self.active_hosts += 1
                switches = 2  # off->active is costlier
        elif action == 2 and self.active_hosts > self.config.min_active_hosts:
            self.active_hosts -= 1
            self.sleeping_hosts += 1
            switches = 1
        elif action == 3 and self.dvfs_idx < len(self.config.dvfs_levels) - 1:
            self.dvfs_idx += 1
        elif action == 4 and self.dvfs_idx > 0:
            self.dvfs_idx -= 1
        elif action == 5 and self.sleeping_hosts > 0:
            self.sleeping_hosts -= 1
            self.off_hosts += 1
            switches = 1

        demand = self._get_demand(0)
        power, mean_util, host_utils, migrations = self._compute_power(demand)
        facility_power, pue = self._facility_power(power, mean_util)
        self._update_thermal_state(host_utils)
        wear = self._hardware_wear()
        sla = self._sla_violation(demand)

        normalized_energy = facility_power / (self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2 * 2.0)
        utilization_bonus = min(mean_util, 1.0)

        reward = -(
            self.config.reward_w_energy * normalized_energy
            + self.config.reward_w_sla * sla
            + self.config.reward_w_switch * switches
            + self.config.reward_w_migration * migrations / max(1, len(host_utils))
        ) + self.config.reward_w_util * utilization_bonus

        info = {
            "demand": demand,
            "it_power": power,
            "power": facility_power,
            "facility_power": facility_power,
            "pue": pue,
            "sla_violation": sla,
            "active_hosts": self.active_hosts,
            "sleep_hosts": self.sleeping_hosts,
            "off_hosts": self.off_hosts,
            "dvfs": self.dvfs,
            "mean_util": mean_util,
            "switches": switches,
            "migrations": migrations,
            "mean_temperature_c": float(np.mean(self.host_temperatures)),
            "hardware_wear_index": wear,
            "host_utils": host_utils.copy(),
            "sla": sla,
        }
        self.trace.append(info)

        self.last_power = power
        self.last_facility_power = facility_power
        self.last_pue = pue
        self.last_sla = sla
        self.last_switches = switches
        self.last_migrations = migrations
        self.last_hardware_wear = wear
        self.step_idx += 1

        terminated = self.step_idx >= self.config.episode_length
        truncated = False
        obs = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.step_idx} active={self.active_hosts} sleep={self.sleeping_hosts} off={self.off_hosts} "
            f"dvfs={self.dvfs:.2f} pue={self.last_pue:.2f} facility_power={self.last_facility_power:.2f} "
            f"temp={np.mean(self.host_temperatures):.1f}C sla={self.last_sla:.4f}"
        )


def load_workload_csv(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    if "demand" not in df.columns:
        raise ValueError("CSV phải có cột 'demand'.")
    values = df["demand"].astype(float).to_numpy()
    return np.clip(values, 0.0, 2.0)
