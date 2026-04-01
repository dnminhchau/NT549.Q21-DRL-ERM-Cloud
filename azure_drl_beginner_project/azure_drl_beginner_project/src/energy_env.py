from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    max_hosts: int = 8
    min_active_hosts: int = 1
    min_sleep_hosts: int = 0
    episode_length: int = 288
    dvfs_levels: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
    host_nominal_capacity: float = 1.0 / 8.0

    p_idle: float = 80.0
    p_peak: float = 200.0
    p_sleep: float = 10.0
    p_off: float = 1.0
    host_switch_cost: float = 8.0
    migration_cost: float = 1.5

    reward_w_energy: float = 1.0
    reward_w_sla: float = 4.0
    reward_w_switch: float = 0.15
    reward_w_migration: float = 0.05
    reward_w_util: float = 0.25
    reward_w_temp: float = 0.10
    reward_w_lifetime: float = 0.05

    base_pue: float = 1.18
    cooling_alpha: float = 0.08
    cooling_beta: float = 0.18
    cooling_fixed_power: float = 25.0

    ambient_temp_c: float = 24.0
    temp_idle_c: float = 33.0
    temp_rise_per_util: float = 30.0
    temp_rise_per_dvfs: float = 10.0
    temp_smoothing: float = 0.30
    reference_temp_c: float = 35.0
    max_safe_temp_c: float = 80.0
    aging_temp_threshold_c: float = 45.0
    aging_per_step_base: float = 1.0
    aging_temp_factor: float = 0.03

    vm_unit_demand: float = 0.05
    seed: int = 42




class CloudEnergyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    STATUS_OFF = 0
    STATUS_SLEEP = 1
    STATUS_ACTIVE = 2

    ACTION_KEEP = 0
    ACTION_WAKE_ONE = 1
    ACTION_SLEEP_ONE = 2
    ACTION_DVFS_UP = 3
    ACTION_DVFS_DOWN = 4
    ACTION_POWER_OFF_ONE = 5
    ACTION_BOOT_ONE = 6

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
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=np.zeros(14, dtype=np.float32),
            high=np.full(14, 2.5, dtype=np.float32),
            dtype=np.float32,
        )
        self.start_idx = 0
        self.step_idx = 0
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self.host_status = np.zeros(self.config.max_hosts, dtype=np.int8)
        self.host_loads = np.zeros(self.config.max_hosts, dtype=np.float32)
        self.host_temps = np.full(self.config.max_hosts, self.config.ambient_temp_c, dtype=np.float32)
        self.host_age = np.zeros(self.config.max_hosts, dtype=np.float32)
        self.prev_assignment: dict[int, int] = {}
       

        self.last_power_it = 0.0
        self.last_power_total = 0.0
        self.last_pue = 1.0
        self.last_sla = 0.0
        self.last_switches = 0
        self.last_migrations = 0
        self.last_temp = self.config.ambient_temp_c
        self.last_lifetime_penalty = 0.0
        self.trace: list[dict[str, Any]] = []
    @property
    def dvfs(self) -> float:
        return self.config.dvfs_levels[self.dvfs_idx]
    @property
    def active_hosts(self) -> int:
        return int(np.sum(self.host_status == self.STATUS_ACTIVE))

    @property
    def sleep_hosts(self) -> int:
        return int(np.sum(self.host_status == self.STATUS_SLEEP))

    @property
    def off_hosts(self) -> int:
        return int(np.sum(self.host_status == self.STATUS_OFF))
    
    def _get_demand(self, offset: int = 0) -> float:
        idx = min(self.start_idx + self.step_idx + offset, self.workload.size - 1)
        return float(self.workload[idx])

    def _cluster_capacity(self) -> float:
        return self.active_hosts * self.config.host_nominal_capacity * self.dvfs
    def _active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_ACTIVE)

    def _sleep_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_SLEEP)

    def _off_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_OFF)

    def _reset_hosts(self):
        self.host_status[:] = self.STATUS_OFF
        initial_active = max(2, self.config.max_hosts // 2)
        active_idx = np.arange(initial_active)
        sleep_idx = np.arange(initial_active, self.config.max_hosts)
        self.host_status[active_idx] = self.STATUS_ACTIVE
        self.host_status[sleep_idx] = self.STATUS_SLEEP
        self.host_loads[:] = 0.0
        self.host_temps[:] = self.config.ambient_temp_c
        self.host_age[:] = 0.0
        self.prev_assignment = {}
        

    def _apply_action(self, action: int) -> tuple[int, int]:
        switches = 0
        wake_from_off = 0

        if action == self.ACTION_WAKE_ONE:
            sleep_idx = self._sleep_indices()
            if sleep_idx.size > 0:
                self.host_status[sleep_idx[0]] = self.STATUS_ACTIVE
                switches += 1
            else:
                off_idx = self._off_indices()
                if off_idx.size > 0:
                    self.host_status[off_idx[0]] = self.STATUS_ACTIVE
                    switches += 1
                    wake_from_off += 1

        elif action == self.ACTION_SLEEP_ONE:
            active_idx = self._active_indices()
            if active_idx.size > self.config.min_active_hosts:
                target = int(active_idx[-1])
                self.host_status[target] = self.STATUS_SLEEP
                self.host_loads[target] = 0.0
                switches += 1

        elif action == self.ACTION_DVFS_UP and self.dvfs_idx < len(self.config.dvfs_levels) - 1:
            self.dvfs_idx += 1

        elif action == self.ACTION_DVFS_DOWN and self.dvfs_idx > 0:
            self.dvfs_idx -= 1

        elif action == self.ACTION_POWER_OFF_ONE:
            sleep_idx = self._sleep_indices()
            if sleep_idx.size > self.config.min_sleep_hosts:
                target = int(sleep_idx[-1])
                self.host_status[target] = self.STATUS_OFF
                self.host_loads[target] = 0.0
                switches += 1

        elif action == self.ACTION_BOOT_ONE:
            off_idx = self._off_indices()
            if off_idx.size > 0:
                self.host_status[off_idx[0]] = self.STATUS_SLEEP
                switches += 1
                wake_from_off += 1

        return switches, wake_from_off

    def _build_vm_demands(self, demand: float) -> np.ndarray:
        if demand <= 0:
            return np.zeros(0, dtype=np.float32)
        n_vms = max(1, int(np.ceil(demand / max(self.config.vm_unit_demand, 1e-6))))
        base = demand / n_vms
        return np.full(n_vms, base, dtype=np.float32)

    

    def _pack_vms(self, vm_demands: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[dict[str, int | float]], int]:
        active_idx = self._active_indices()
        host_loads = np.zeros(self.config.max_hosts, dtype=np.float32)
        if active_idx.size == 0 or vm_demands.size == 0:
            return host_loads, {}, [], 0

        host_capacity = self.config.host_nominal_capacity * self.dvfs
        remaining = {int(h): host_capacity for h in active_idx.tolist()}
        assignment: dict[int, int] = {}
        migration_plan: list[dict[str, int | float]] = []
        migrations = 0

        for vm_id, demand_unit in enumerate(sorted(vm_demands.tolist(), reverse=True)):
            previous_host = self.prev_assignment.get(vm_id)
            candidate_hosts = sorted(
                active_idx.tolist(),
                key=lambda h: (remaining[int(h)] - demand_unit if remaining[int(h)] >= demand_unit else 1e9, h),
            )

            placed = False
            for host in candidate_hosts:
                host = int(host)
                if remaining[host] + 1e-9 >= demand_unit:
                    remaining[host] -= demand_unit
                    host_loads[host] += demand_unit
                    assignment[vm_id] = host
                    if previous_host is not None and previous_host != host:
                        migrations += 1
                        migration_plan.append({
                            "vm_id": int(vm_id),
                            "from": int(previous_host),
                            "to": host,
                            "demand": float(demand_unit),
                        })
                    placed = True
                    break

            if not placed:
                fullest = int(sorted(active_idx.tolist(), key=lambda h: host_loads[int(h)], reverse=True)[0])
                host_loads[fullest] += demand_unit
                assignment[vm_id] = fullest
                if previous_host is not None and previous_host != fullest:
                    migrations += 1
                    migration_plan.append({
                        "vm_id": int(vm_id),
                        "from": int(previous_host),
                        "to": fullest,
                        "demand": float(demand_unit),
                    })

        return host_loads, assignment, migration_plan, migrations
    def _update_temperatures(self):
        for i in range(self.config.max_hosts):
            if self.host_status[i] == self.STATUS_ACTIVE:
                util = min(
                    self.host_loads[i] / max(self.config.host_nominal_capacity * self.dvfs, 1e-8),
                    1.5,
                )
                target_temp = (
                    self.config.temp_idle_c
                    + self.config.temp_rise_per_util * min(util, 1.0)
                    + self.config.temp_rise_per_dvfs * max(self.dvfs - 0.6, 0.0)
                )
            elif self.host_status[i] == self.STATUS_SLEEP:
                target_temp = self.config.ambient_temp_c + 4.0
            else:
                target_temp = self.config.ambient_temp_c + 1.0

            self.host_temps[i] = (
                (1.0 - self.config.temp_smoothing) * self.host_temps[i]
                + self.config.temp_smoothing * target_temp
            )

    def _update_age(self) -> float:
        lifetime_penalty = 0.0
        for i in range(self.config.max_hosts):
            temp = float(self.host_temps[i])
            if self.host_status[i] == self.STATUS_OFF:
                self.host_age[i] += 0.15 * self.config.aging_per_step_base
                continue

            thermal_excess = max(0.0, temp - self.config.aging_temp_threshold_c)
            aging_increment = self.config.aging_per_step_base * (1.0 + self.config.aging_temp_factor * thermal_excess)
            self.host_age[i] += aging_increment
            lifetime_penalty += max(0.0, aging_increment - self.config.aging_per_step_base)
        return float(lifetime_penalty)
    
    def _compute_power(self, demand: float) -> tuple[float, float, float, np.ndarray, float, float]:
        active_idx = self._active_indices()
        host_utils = np.zeros(self.config.max_hosts, dtype=np.float32)
        it_power = 0.0
        mean_util = 0.0

        if active_idx.size > 0:
            cap = max(self.config.host_nominal_capacity * self.dvfs, 1e-8)
            utils = np.clip(self.host_loads[active_idx] / cap, 0.0, 1.5)
            host_utils[active_idx] = utils.astype(np.float32)
            mean_util = float(np.mean(utils)) if utils.size else 0.0

            active_power = (
                self.config.p_idle
                + (self.config.p_peak - self.config.p_idle)
                * np.minimum(utils, 1.0)
                * (self.dvfs ** 2)
            )
            it_power += float(np.sum(active_power))

        it_power += self.sleep_hosts * self.config.p_sleep + self.off_hosts * self.config.p_off

        avg_active_temp = (
            float(np.mean(self.host_temps[active_idx]))
            if active_idx.size > 0
            else self.config.ambient_temp_c
        )

        cooling_multiplier = (
            self.config.base_pue
            + self.config.cooling_alpha * mean_util
            + self.config.cooling_beta * max(0.0, (avg_active_temp - self.config.reference_temp_c) / 30.0)
        )
        cooling_multiplier = max(cooling_multiplier, 1.0)

        total_power = it_power * cooling_multiplier + self.config.cooling_fixed_power
        estimated_pue = total_power / max(it_power, 1e-8)
        cooling_power = max(0.0, total_power - it_power)

        return (
            float(it_power),
            float(total_power),
            float(mean_util),
            host_utils,
            float(estimated_pue),
            float(cooling_power),
        )
    def _sla_violation(self, demand: float) -> float:
        cap = self._cluster_capacity()
        if cap <= 0:
            return 1.0
        return float(max(0.0, demand - cap) / max(demand, 1e-8))

    def _observation(self) -> np.ndarray:
        demand_now = self._get_demand(0)
        demand_next = self._get_demand(1)
        active_ratio = self.active_hosts / self.config.max_hosts
        sleep_ratio = self.sleep_hosts / self.config.max_hosts
        off_ratio = self.off_hosts / self.config.max_hosts
        avg_temp = float(np.mean(self.host_temps))
        temp_ratio = avg_temp / max(self.config.max_safe_temp_c, 1.0)
        mean_host_age = float(np.mean(self.host_age)) / 1000.0

        obs = np.array(
            [
                demand_now,
                demand_next,
                active_ratio,
                self.dvfs / max(self.config.dvfs_levels),
                self.last_power_total / (self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2 + self.config.cooling_fixed_power),
                self.last_sla,
                float(np.mean(self.host_loads[self._active_indices()])) if self.active_hosts > 0 else 0.0,
                float(np.std(self.host_loads[self._active_indices()])) if self.active_hosts > 0 else 0.0,
                sleep_ratio,
                off_ratio,
                self.last_pue / 3.0,
                temp_ratio,
                mean_host_age,
                self.last_migrations / max(self.config.max_hosts, 1),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = self.workload.size - self.config.episode_length - 1
        self.start_idx = int(self.rng.integers(0, max(1, max_start + 1)))
        self.step_idx = 0
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)
        self._reset_hosts()

        demand = self._get_demand(0)
        vm_demands = self._build_vm_demands(demand)
        self.host_loads, self.prev_assignment, _, self.last_migrations = self._pack_vms(vm_demands)
        self._update_temperatures()
        self.last_lifetime_penalty = self._update_age()
        self.last_power_it, self.last_power_total, _, _, self.last_pue, _ = self._compute_power(demand)
        self.last_sla = self._sla_violation(demand)
        self.last_switches = 0
        self.last_temp = float(np.mean(self.host_temps))
        self.trace = []
        return self._observation(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"

        switches, wake_from_off = self._apply_action(int(action))
        demand = self._get_demand(0)

        vm_demands = self._build_vm_demands(demand)
        self.host_loads, assignment, migration_plan, migrations = self._pack_vms(vm_demands)
        self.prev_assignment = assignment

        self._update_temperatures()
        lifetime_penalty = self._update_age()

        power_it, power_total, mean_util, host_utils, pue, cooling_power = self._compute_power(demand)
        sla = self._sla_violation(demand)

        normalized_energy = power_total / (
            self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2
            + self.config.cooling_fixed_power
        )
        utilization_bonus = min(mean_util, 1.0)
        avg_temp = float(np.mean(self.host_temps))
        thermal_penalty = max(
            0.0,
            avg_temp - self.config.reference_temp_c
        ) / max(
            self.config.max_safe_temp_c - self.config.reference_temp_c,
            1.0
        )

        switch_penalty = switches * self.config.host_switch_cost
        migration_penalty = migrations * self.config.migration_cost
        reward = -(
            self.config.reward_w_energy * normalized_energy
            + self.config.reward_w_sla * sla
            + self.config.reward_w_switch * switch_penalty
            + self.config.reward_w_migration * migration_penalty
            + self.config.reward_w_temp * thermal_penalty
            + self.config.reward_w_lifetime * lifetime_penalty / max(self.config.max_hosts, 1)
        ) + self.config.reward_w_util * utilization_bonus
        if sla > 0.2:
            reward -= 10.0

        info = {
            "demand": demand,
            "power_it": power_it,
            "power": power_total,
            "power_total": power_total,
            "pue": pue,
            "sla_violation": sla,
            "active_hosts": self.active_hosts,
            "sleep_hosts": self.sleep_hosts,
            "off_hosts": self.off_hosts,
            "dvfs": self.dvfs,
            "mean_util": mean_util,
            "switches": switches,
            "wake_from_off": wake_from_off,
            "migrations": migrations,
            "avg_temp": avg_temp,
            "max_temp": float(np.max(self.host_temps)),
            "thermal_penalty": thermal_penalty,
            "lifetime_penalty": float(lifetime_penalty),
            "mean_host_age": float(np.mean(self.host_age)),
            "sla": sla,
            "migration_plan": migration_plan,
            "cooling_power": cooling_power,
            "host_loads": self.host_loads.copy().tolist(),
            "host_temps": self.host_temps.copy().tolist(),
            "host_status": self.host_status.copy().tolist(),
            "host_utils": host_utils.copy().tolist(),
        }
        self.trace.append(info)

        self.last_power_it = power_it
        self.last_power_total = power_total
        self.last_pue = pue
        self.last_sla = sla
        self.last_switches = switches
        self.last_migrations = migrations
        self.last_temp = avg_temp
        self.last_lifetime_penalty = lifetime_penalty
        self.step_idx += 1

        terminated = self.step_idx >= self.config.episode_length
        truncated = False
        obs = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.step_idx} active={self.active_hosts} sleep={self.sleep_hosts} off={self.off_hosts} "
            f"dvfs={self.dvfs:.2f} power_total={self.last_power_total:.2f} pue={self.last_pue:.3f} "
            f"temp={self.last_temp:.2f} sla={self.last_sla:.4f} migrations={self.last_migrations}"
        )


def load_workload_csv(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    if "demand" not in df.columns:
        raise ValueError("CSV phải có cột 'demand'.")
    values = df["demand"].astype(float).to_numpy()
    return np.clip(values, 0.0, 2.0)
