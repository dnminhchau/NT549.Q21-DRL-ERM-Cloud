from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class EnvConfig:
    use_safety_guard: bool = True
    capacity_safety_margin: float = 1.05
    high_load_guard_threshold: float = 0.85
    reward_w_guard_override: float = 0.05

    reward_w_migration_count: float = 0.35
    migration_count_clip: float = 0.50

    reward_w_hotspot: float = 0.12
    hotspot_temp_threshold_c: float = 60.0

    max_hosts: int = 8
    min_active_hosts: int = 1
    min_sleep_hosts: int = 0
    episode_length: int = 160
    dvfs_levels: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
    host_nominal_capacity: float = 1.0 / 7.0

    p_idle: float = 80.0
    p_peak: float = 200.0
    p_sleep: float = 10.0

    p_off: float = 0.3
    host_switch_cost: float = 5.0

    migration_cost: float = 1.5

    priority_migration_multiplier: float = 3.0   
    memory_migration_scale: float = 0.01          
    migration_max_size_factor: float = 5.0        

    reward_w_energy: float = 1.60
    reward_w_sla: float = 4.50
    reward_w_switch: float = 0.08
    reward_w_migration: float = 0.10
    reward_w_latency: float = 0.35
    reward_w_util: float = 0.18
    reward_w_temp: float = 0.18
    reward_w_lifetime: float = 0.08
    reward_w_overprovision: float = 0.90
    reward_w_active_excess: float = 1.10
    reward_w_sleep_excess: float = 0.30
    reward_w_off_bonus: float = 0.30
    reward_w_pue: float = 0.40
    pue_norm_max: float = 2.0

    reward_w_dvfs: float = 0.55
    reward_w_dvfs_mismatch: float = 0.45
    reward_w_sticky_config: float = 0.03

    target_host_util: float = 0.82
    reserve_sleep_hosts: int = 1

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

    vm_unit_demand: float = 0.10


    vm_snapshot_path: str | None = None
    vm_snapshot_max_chunks_per_type: int = 8
    vm_snapshot_min_group_demand: float = 1e-5
    obs_clip_high: float = 2.5
    obs_memory_percentile: float = 95.0
    latency_util_clip: float = 0.95
    latency_cap: float = 10.0

    sla_extra_threshold_1: float = 0.05
    sla_extra_penalty_1: float = 1.00

    sla_extra_threshold_2: float = 0.15
    sla_extra_penalty_2: float = 2.50

    sla_penalty_growth_threshold: float = 0.10
    sla_penalty_growth_factor: float = 5.00

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
        self.vm_snapshots = self._load_vm_snapshots(self.config.vm_snapshot_path)
        self.uses_vm_snapshots = bool(self.vm_snapshots)
        self.obs_memory_scale = self._estimate_memory_scale()

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=np.zeros(17, dtype=np.float32),
            high=np.full(17, self.config.obs_clip_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.start_idx = 0
        self.step_idx = 0
        self.dvfs_idx = min(2, len(self.config.dvfs_levels) - 1)

        self.host_status = np.zeros(self.config.max_hosts, dtype=np.int8)
        self.host_loads = np.zeros(self.config.max_hosts, dtype=np.float32)
        self.host_temps = np.full(
            self.config.max_hosts,
            self.config.ambient_temp_c,
            dtype=np.float32,
        )
        self.host_age = np.zeros(self.config.max_hosts, dtype=np.float32)
        self.prev_assignment: dict[int, int] = {}

        self.prev_active_hosts = 0
        self.prev_dvfs = float(self.dvfs)
        self.same_config_steps = 0

        self.last_power_it = 0.0
        self.last_power_total = 0.0
        self.last_pue = 1.0
        self.last_sla = 0.0
        self.last_switches = 0
        self.last_migrations = 0
        self.last_vm_count = 0
        self.last_temp = self.config.ambient_temp_c
        self.last_lifetime_penalty = 0.0
        self.last_priority1_ratio = 0.0
        self.last_avg_vm_memory = 0.0
        self.last_migration_penalty_weighted = 0.0
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

    def _host_capacity(self, dvfs: float | None = None) -> float:
        dvfs_value = self.dvfs if dvfs is None else float(dvfs)
        return self.config.host_nominal_capacity * dvfs_value

    def _desired_active_hosts(self, demand: float, dvfs: float | None = None) -> int:
        per_host_effective = self._host_capacity(dvfs) * self.config.target_host_util
        if per_host_effective <= 1e-8:
            return self.config.max_hosts
        needed = int(np.ceil(demand / per_host_effective))
        return int(np.clip(needed, self.config.min_active_hosts, self.config.max_hosts))

    def _desired_sleep_hosts(self, demand_now: float, demand_next: float) -> int:
        reserve = self.config.reserve_sleep_hosts if demand_next > demand_now + 0.03 else 0
        max_allowable = max(0, self.config.max_hosts - self._desired_active_hosts(demand_now))
        reserve = min(reserve, max_allowable)
        return int(max(self.config.min_sleep_hosts, reserve))

    def _active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_ACTIVE)

    def _sleep_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_SLEEP)

    def _off_indices(self) -> np.ndarray:
        return np.flatnonzero(self.host_status == self.STATUS_OFF)

    def _reset_hosts(self, demand: float | None = None):
        self.host_status[:] = self.STATUS_OFF

        if demand is None:
            desired_active = max(self.config.min_active_hosts, 2)
        else:
            desired_active = self._desired_active_hosts(
                float(demand),
                dvfs=self.config.dvfs_levels[self.dvfs_idx],
            )

        initial_active = min(desired_active, self.config.max_hosts - 1)
        initial_active = max(self.config.min_active_hosts, initial_active)

        initial_sleep = min(
            max(self.config.reserve_sleep_hosts, self.config.min_sleep_hosts),
            max(0, self.config.max_hosts - initial_active),
        )

        active_idx = np.arange(initial_active)
        sleep_idx = np.arange(initial_active, initial_active + initial_sleep)

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
                self.host_status[int(sleep_idx[0])] = self.STATUS_ACTIVE
                switches += 1

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
                self.host_status[int(off_idx[0])] = self.STATUS_ACTIVE
                switches += 1
                wake_from_off += 1

        return switches, wake_from_off
    

    def _projected_capacity_after_action(self, action: int) -> float:
        projected_active = self.active_hosts
        projected_dvfs_idx = self.dvfs_idx

        if action == self.ACTION_SLEEP_ONE and projected_active > self.config.min_active_hosts:
            projected_active -= 1

        elif action == self.ACTION_POWER_OFF_ONE:
            projected_active = projected_active

        elif action == self.ACTION_DVFS_DOWN and projected_dvfs_idx > 0:
            projected_dvfs_idx -= 1

        elif action in (self.ACTION_WAKE_ONE, self.ACTION_BOOT_ONE):
            projected_active = min(projected_active + 1, self.config.max_hosts)

        elif action == self.ACTION_DVFS_UP and projected_dvfs_idx < len(self.config.dvfs_levels) - 1:
            projected_dvfs_idx += 1

        projected_dvfs = self.config.dvfs_levels[projected_dvfs_idx]
        return projected_active * self.config.host_nominal_capacity * projected_dvfs


    def _safety_guard_action(self, action: int, demand_now: float, demand_next: float) -> int:
        if not self.config.use_safety_guard:
            return action

        risky_actions = {
            self.ACTION_SLEEP_ONE,
            self.ACTION_POWER_OFF_ONE,
            self.ACTION_DVFS_DOWN,
        }

        if action not in risky_actions:
            return action

        required = max(demand_now, demand_next) * self.config.capacity_safety_margin
        projected_capacity = self._projected_capacity_after_action(action)

        if projected_capacity < required:
            return self.ACTION_KEEP

        if max(demand_now, demand_next) >= self.config.high_load_guard_threshold:
            return self.ACTION_KEEP

        return action

    
    
    
    def _load_vm_snapshots(self, path: str | None) -> dict[int, list[dict[str, Any]]]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}

        df = pd.read_csv(p)
        required = {"timestep", "vmTypeId", "priority", "count_active", "core", "memory"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"vm_snapshots.csv thiếu cột: {sorted(missing)}")

        snapshots: dict[int, list[dict[str, Any]]] = {}
        for row in df.itertuples(index=False):
            timestep = int(getattr(row, "timestep"))
            snapshots.setdefault(timestep, []).append(
                {
                    "vmTypeId": int(getattr(row, "vmTypeId")),
                    "priority": int(getattr(row, "priority")),
                    "count_active": int(getattr(row, "count_active")),
                    "core": float(getattr(row, "core")),
                    "memory": float(getattr(row, "memory")),
                    "has_priority1": int(getattr(row, "has_priority1", int(getattr(row, "priority")) == 1)),
                }
            )
        return snapshots

    def _estimate_memory_scale(self) -> float:
        if not self.vm_snapshots:
            return max(self.config.memory_migration_scale, 1e-6)

        memories: list[float] = []
        for rows in self.vm_snapshots.values():
            for row in rows:
                memories.append(float(row.get("memory", 0.0)))
        if not memories:
            return max(self.config.memory_migration_scale, 1e-6)

        percentile = float(np.percentile(memories, self.config.obs_memory_percentile))
        return max(percentile, self.config.memory_migration_scale, 1e-6)

    def _normalize_feature(self, value: float, scale: float) -> float:
        if scale <= 1e-9:
            return 0.0
        norm = value / scale
        return float(np.clip(norm, 0.0, self.config.obs_clip_high))

    def _build_vm_records(self, demand: float) -> list[dict[str, Any]]:
        if demand <= 0:
            return []

        global_timestep = self.start_idx + self.step_idx
        snapshot_rows = self.vm_snapshots.get(global_timestep, [])

        if snapshot_rows:
            raw_total_core = sum(
                max(0.0, float(r["count_active"])) * max(0.0, float(r["core"]))
                for r in snapshot_rows
            )
            if raw_total_core > 1e-12:
                scale = float(demand) / raw_total_core
                records: list[dict[str, Any]] = []
                max_chunks = max(1, int(self.config.vm_snapshot_max_chunks_per_type))
                unit = max(float(self.config.vm_unit_demand), 1e-6)

                for r in snapshot_rows:
                    vm_type = int(r["vmTypeId"])
                    priority = int(r["priority"])
                    count_active = int(r["count_active"])
                    group_demand = float(count_active) * float(r["core"]) * scale
                    group_memory = float(count_active) * float(r["memory"]) * scale
                    if group_demand <= self.config.vm_snapshot_min_group_demand:
                        continue

                    n_chunks = int(np.ceil(group_demand / unit))
                    n_chunks = max(1, min(max_chunks, n_chunks))
                    demand_unit = group_demand / n_chunks
                    memory_unit = group_memory / n_chunks if n_chunks > 0 else 0.0
                    count_unit = count_active / n_chunks if n_chunks > 0 else count_active

                    for chunk_id in range(n_chunks):
                        # Stable ID để so sánh host cũ/mới qua timestep.
                        vm_id = vm_type * 100_000 + priority * 10_000 + chunk_id
                        records.append(
                            {
                                "vm_id": int(vm_id),
                                "vm_type_id": int(vm_type),
                                "priority": int(priority),
                                "demand": float(demand_unit),
                                "memory": float(memory_unit),
                                "represented_count": float(count_unit),
                                "source": "azure_vm_snapshot",
                            }
                        )

                if records:
                    return records

        # Fallback: pseudo-VM cũ nếu chưa có vm_snapshots.csv.
        n_vms = max(1, int(np.ceil(demand / max(self.config.vm_unit_demand, 1e-6))))
        base = demand / n_vms
        return [
            {
                "vm_id": int(i),
                "vm_type_id": -1,
                "priority": 0,
                "demand": float(base),
                "memory": float(base),
                "represented_count": 1.0,
                "source": "fallback_pseudo_vm",
            }
            for i in range(n_vms)
        ]

    def _pack_vm_records(
        self, vm_records: list[dict[str, Any]]
    ) -> tuple[np.ndarray, dict[int, int], list[dict[str, int | float | str]], int, list[dict[str, Any]], dict[str, int]]:
        
        # Migration-aware Best-Fit Consolidation.
        # - Ưu tiên giữ VM ở host cũ nếu host đó active và còn đủ tài nguyên.
        # - Nếu không giữ được, chọn host active có phần tài nguyên dư sau khi đặt là nhỏ nhất.
        # - Nếu VM đổi host so với prev_assignment thì ghi migration event.
        
        active_idx = self._active_indices()
        host_loads = np.zeros(self.config.max_hosts, dtype=np.float32)

        if active_idx.size == 0 or not vm_records:
            return host_loads, {}, [], 0, [], {}

        host_capacity = self.config.host_nominal_capacity * self.dvfs
        remaining = {int(h): float(host_capacity) for h in active_idx}

        # VM lớn và priority cao được xếp trước.
        sorted_records = sorted(
            vm_records,
            key=lambda r: (
                0 if int(r.get("priority", 0)) == 1 else 1,
                -float(r.get("demand", 0.0)),
            ),
        )

        assignment: dict[int, int] = {}
        migration_events: list[dict[str, int | float | str]] = []
        vm_states: list[dict[str, Any]] = []
        migrations = 0

        for record in sorted_records:
            vm_id = int(record["vm_id"])
            demand_unit = float(record["demand"])
            previous_host = self.prev_assignment.get(vm_id)

            chosen_host: int | None = None

            # 1) Migration-aware: giữ VM ở host cũ nếu còn active và đủ chỗ.
            if previous_host is not None and previous_host in remaining:
                if remaining[previous_host] >= demand_unit - 1e-9:
                    chosen_host = int(previous_host)

            # 2) Best-fit: chọn host còn đủ chỗ và dư ít nhất sau khi đặt.
            if chosen_host is None:
                feasible_hosts = [h for h, rem in remaining.items() if rem >= demand_unit - 1e-9]
                if feasible_hosts:
                    chosen_host = min(feasible_hosts, key=lambda h: remaining[h] - demand_unit)
                else:
                    # Nếu VM group quá lớn, đặt vào host còn dư nhiều nhất và cho phép overload nhẹ.
                    chosen_host = max(remaining.keys(), key=lambda h: remaining[h])

            remaining[chosen_host] -= demand_unit
            host_loads[chosen_host] += demand_unit
            assignment[vm_id] = int(chosen_host)

            if previous_host is not None and previous_host != chosen_host:
                migrations += 1
                migration_events.append(
                    {
                        "vm_id": int(vm_id),
                        "vm_type_id": int(record.get("vm_type_id", -1)),
                        "priority": int(record.get("priority", 0)),
                        "from": int(previous_host),
                        "to": int(chosen_host),
                        "demand": float(demand_unit),
                        "memory": float(record.get("memory", 0.0)),
                        "represented_count": float(record.get("represented_count", 1.0)),
                        "source": str(record.get("source", "unknown")),
                    }
                )

            vm_states.append(
                {
                    "vm_id": int(vm_id),
                    "vm_type_id": int(record.get("vm_type_id", -1)),
                    "priority": int(record.get("priority", 0)),
                    "host_id": int(chosen_host),
                    "demand": float(demand_unit),
                    "memory": float(record.get("memory", 0.0)),
                    "represented_count": float(record.get("represented_count", 1.0)),
                    "source": str(record.get("source", "unknown")),
                }
            )

        vm_to_host_map = {str(vm_id): int(host) for vm_id, host in assignment.items()}
        return host_loads, assignment, migration_events, migrations, vm_states, vm_to_host_map

    def _update_vm_stats(self, vm_states: list[dict[str, Any]]):
        if not vm_states:
            self.last_vm_count = 0
            self.last_priority1_ratio = 0.0
            self.last_avg_vm_memory = 0.0
            return

        self.last_vm_count = len(vm_states)
        total_demand = 0.0
        priority1_demand = 0.0
        memories: list[float] = []
        for vm in vm_states:
            demand = float(vm.get("demand", 0.0))
            total_demand += demand
            if int(vm.get("priority", 0)) == 1:
                priority1_demand += demand
            memories.append(float(vm.get("memory", 0.0)))

        self.last_priority1_ratio = (
            priority1_demand / total_demand if total_demand > 1e-8 else 0.0
        )
        avg_memory = float(np.mean(memories)) if memories else 0.0
        self.last_avg_vm_memory = self._normalize_feature(avg_memory, self.obs_memory_scale)

    def _host_state_snapshot(self, host_utils: np.ndarray | None = None) -> list[dict[str, Any]]:
        status_name = {
            self.STATUS_OFF: "off",
            self.STATUS_SLEEP: "sleep",
            self.STATUS_ACTIVE: "active",
        }
        rows: list[dict[str, Any]] = []
        for i in range(self.config.max_hosts):
            util = 0.0 if host_utils is None else float(host_utils[i])
            rows.append(
                {
                    "host_id": int(i),
                    "status": status_name.get(int(self.host_status[i]), "unknown"),
                    "load": float(self.host_loads[i]),
                    "utilization": util,
                    "temperature": float(self.host_temps[i]),
                    "age": float(self.host_age[i]),
                }
            )
        return rows

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
            aging_increment = self.config.aging_per_step_base * (
                1.0 + self.config.aging_temp_factor * thermal_excess
            )
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
            + self.config.cooling_beta * max(
                0.0, (avg_active_temp - self.config.reference_temp_c) / 30.0
            )
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

    def _dvfs_penalty(self, demand: float, mean_util: float) -> tuple[float, float, float]:
        dvfs_min = float(min(self.config.dvfs_levels))
        dvfs_max = float(max(self.config.dvfs_levels))
        dvfs_norm = (self.dvfs - dvfs_min) / max(dvfs_max - dvfs_min, 1e-8)

        if self.active_hosts <= 0:
            desired_dvfs = dvfs_min
        else:
            desired_dvfs = demand / max(
                self.active_hosts * self.config.host_nominal_capacity * self.config.target_host_util,
                1e-8,
            )
            desired_dvfs = float(np.clip(desired_dvfs, dvfs_min, dvfs_max))

        desired_norm = (desired_dvfs - dvfs_min) / max(dvfs_max - dvfs_min, 1e-8)

        high_dvfs_penalty = dvfs_norm ** 2
        mismatch_penalty = abs(dvfs_norm - desired_norm)

        low_load_factor = float(np.clip((0.75 - demand) / 0.75, 0.0, 1.0))
        low_util_factor = float(np.clip((0.80 - mean_util) / 0.80, 0.0, 1.0))
        extra_idle_penalty = dvfs_norm * max(low_load_factor, low_util_factor)

        total_penalty = (
            0.45 * high_dvfs_penalty
            + 0.35 * mismatch_penalty
            + 0.20 * extra_idle_penalty
        )
        return float(total_penalty), float(desired_dvfs), float(mismatch_penalty)

    def _observation(self) -> np.ndarray:
        demand_now = self._get_demand(0)
        demand_next = self._get_demand(1)

        active_ratio = self.active_hosts / self.config.max_hosts
        sleep_ratio = self.sleep_hosts / self.config.max_hosts
        off_ratio = self.off_hosts / self.config.max_hosts

        avg_temp = float(np.mean(self.host_temps))
        temp_ratio = avg_temp / max(self.config.max_safe_temp_c, 1.0)
        mean_host_age = float(np.mean(self.host_age)) / 1000.0

        active_idx = self._active_indices()
        if active_idx.size > 0:
            active_load_mean = float(np.mean(self.host_loads[active_idx]))
            active_load_std = float(np.std(self.host_loads[active_idx]))
        else:
            active_load_mean = 0.0
            active_load_std = 0.0

        obs = np.array(
            [
                demand_now,
                demand_next,
                active_ratio,
                self.dvfs / max(self.config.dvfs_levels),
                self.last_power_total / (
                    self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2
                    + self.config.cooling_fixed_power
                ),
                self.last_sla,
                active_load_mean,
                active_load_std,
                sleep_ratio,
                off_ratio,
                self.last_pue / 3.0,
                temp_ratio,
                mean_host_age,
                self._normalize_feature(
                    float(self.last_migrations), float(max(self.last_vm_count, 1))
                ),

                self.last_priority1_ratio,           
                self.last_avg_vm_memory,             
                self.last_migration_penalty_weighted,  
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

        demand = self._get_demand(0)
        self._reset_hosts(demand)

        self.prev_active_hosts = int(self.active_hosts)
        self.prev_dvfs = float(self.dvfs)
        self.same_config_steps = 0

        vm_records = self._build_vm_records(demand)
        (
            self.host_loads,
            self.prev_assignment,
            _,
            self.last_migrations,
            vm_states,
            _,
        ) = self._pack_vm_records(vm_records)
        self._update_vm_stats(vm_states)
        self.last_migration_penalty_weighted = 0.0

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

        action = int(action)
        raw_action = action

        demand_now_for_guard = self._get_demand(0)
        demand_next_for_guard = self._get_demand(1)
        action = self._safety_guard_action(action, demand_now_for_guard, demand_next_for_guard)
        guard_changed_action = int(action != raw_action)

        invalid_action_penalty = 0.0
        if action == self.ACTION_WAKE_ONE and self.sleep_hosts == 0:
            invalid_action_penalty = 0.05
        elif action == self.ACTION_SLEEP_ONE and self.active_hosts <= self.config.min_active_hosts:
            invalid_action_penalty = 0.05
        elif action == self.ACTION_DVFS_UP and self.dvfs_idx >= len(self.config.dvfs_levels) - 1:
            invalid_action_penalty = 0.05
        elif action == self.ACTION_DVFS_DOWN and self.dvfs_idx <= 0:
            invalid_action_penalty = 0.05
        elif action == self.ACTION_POWER_OFF_ONE and self.sleep_hosts <= self.config.min_sleep_hosts:
            invalid_action_penalty = 0.05
        elif action == self.ACTION_BOOT_ONE and self.off_hosts == 0:
            invalid_action_penalty = 0.05

        switches, wake_from_off = self._apply_action(action)

        demand = self._get_demand(0)

        vm_records = self._build_vm_records(demand)
        self.host_loads, assignment, migration_events, migrations, vm_states, vm_to_host_map = self._pack_vm_records(vm_records)
        self.prev_assignment = assignment
        self._update_vm_stats(vm_states)

        self._update_temperatures()
        lifetime_penalty = self._update_age()

        power_it, power_total, mean_util, host_utils, pue, cooling_power = self._compute_power(demand)
        sla = self._sla_violation(demand)

        if int(self.active_hosts) == int(self.prev_active_hosts) and float(self.dvfs) == float(self.prev_dvfs):
            self.same_config_steps += 1
        else:
            self.same_config_steps = 0

        sticky_config_penalty = min(self.same_config_steps / 20.0, 1.0)

        self.prev_active_hosts = int(self.active_hosts)
        self.prev_dvfs = float(self.dvfs)

        normalized_energy = power_total / (
            self.config.max_hosts * self.config.p_peak * max(self.config.dvfs_levels) ** 2
            + self.config.cooling_fixed_power
        )

        utilization_bonus = min(mean_util, 1.0)
        balanced_util_bonus = max(0.0, 1.0 - abs(mean_util - 0.78) / 0.78)

        avg_temp = float(np.mean(self.host_temps))

        max_temp = float(np.max(self.host_temps))
        hotspot_penalty = max(0.0, max_temp - self.config.hotspot_temp_threshold_c) / max(
            self.config.max_safe_temp_c - self.config.hotspot_temp_threshold_c,
            1.0,
        )

        thermal_penalty = max(0.0, avg_temp - self.config.reference_temp_c) / max(
            self.config.max_safe_temp_c - self.config.reference_temp_c, 1.0
        )

        cap = self._cluster_capacity()
        spare_ratio = max(0.0, cap - demand) / max(cap, 1e-8)
        overprovision_penalty = max(0.0, spare_ratio - 0.05)
        if cap <= 1e-8:
            util_ratio = 1.0
            latency_penalty = self.config.latency_cap
        else:
            util_ratio = max(demand / cap, 0.0)
            if util_ratio >= 1.0:
                latency_penalty = self.config.latency_cap
            else:
                util_clipped = min(util_ratio, self.config.latency_util_clip)
                latency_penalty = util_clipped / max(1.0 - util_clipped, 1e-8)
                latency_penalty = min(latency_penalty, self.config.latency_cap)

        dvfs_penalty, desired_dvfs, dvfs_mismatch = self._dvfs_penalty(demand, mean_util)

        demand_next = self._get_demand(1)
        desired_active = self._desired_active_hosts(demand)
        desired_sleep = self._desired_sleep_hosts(demand, demand_next)

        active_excess_penalty = max(0.0, self.active_hosts - desired_active) / max(
            self.config.max_hosts, 1
        )
        sleep_excess_penalty = max(0.0, self.sleep_hosts - desired_sleep) / max(
            self.config.max_hosts, 1
        )

        wake_penalty = 0.0
        if action in (self.ACTION_WAKE_ONE, self.ACTION_BOOT_ONE) and self.active_hosts > desired_active:
            wake_penalty = max(spare_ratio, active_excess_penalty)

        off_bonus = 0.0
        if sla < 0.01 and spare_ratio > 0.05 and self.off_hosts > 0:
            off_bonus = self.off_hosts / max(self.config.max_hosts, 1)
            if demand_next > demand + 0.08:
                off_bonus *= 0.5

        power_off_bonus = 0.0
        if action == self.ACTION_POWER_OFF_ONE and sla < 0.01 and spare_ratio > 0.08:
            power_off_bonus = 0.05

        migration_penalty = 0.0
        for event in migration_events:
            priority_factor = (
                self.config.priority_migration_multiplier
                if int(event.get("priority", 0)) == 1
                else 1.0
                )
            mem = float(event.get("memory", self.config.memory_migration_scale))
            size_factor = min(
                mem / max(self.config.memory_migration_scale, 1e-9),
                self.config.migration_max_size_factor,
                )
            migration_penalty += self.config.migration_cost * priority_factor * size_factor
        max_penalty_per_vm = (
            self.config.migration_cost
            * max(self.config.priority_migration_multiplier, 1.0)
            * max(self.config.migration_max_size_factor, 1.0)
        )
        migration_scale = max_penalty_per_vm * max(len(vm_states), 1)
        self.last_migration_penalty_weighted = self._normalize_feature(
            migration_penalty, migration_scale
        )

        sla_penalty = sla * (
            1.0
            + self.config.sla_penalty_growth_factor
            * max(0.0, sla - self.config.sla_penalty_growth_threshold)
        )
        latency_clipped = min(latency_penalty / self.config.latency_cap, 1.0)

        pue_norm_max = max(self.config.pue_norm_max, 1.01)
        pue_penalty = max(0.0, pue - 1.0) / max(pue_norm_max - 1.0, 1e-8)
        pue_penalty = min(pue_penalty, 1.0)

        combined_util_bonus = (
            0.5 * min(mean_util, 1.0)
            + 0.5 * max(0.0, 1.0 - abs(mean_util - 0.78) / 0.78)
        )

        migration_penalty_norm = self.last_migration_penalty_weighted
        migration_count_ratio = migrations / max(len(vm_states), 1)
        migration_count_penalty = min(migration_count_ratio, self.config.migration_count_clip)

        lifetime_penalty_norm = lifetime_penalty / max(self.config.max_hosts, 1)

        reward = (
            -(
                self.config.reward_w_energy * normalized_energy
                + self.config.reward_w_sla * sla_penalty
                + self.config.reward_w_latency * latency_clipped
                + self.config.reward_w_switch * (switches / max(self.config.max_hosts, 1))
                + self.config.reward_w_migration * migration_penalty_norm
                + self.config.reward_w_temp * thermal_penalty
                + self.config.reward_w_hotspot * hotspot_penalty
                + self.config.reward_w_lifetime * lifetime_penalty_norm
                + self.config.reward_w_overprovision * overprovision_penalty
                + self.config.reward_w_dvfs * dvfs_penalty
                + self.config.reward_w_dvfs_mismatch * dvfs_mismatch
                + self.config.reward_w_active_excess * active_excess_penalty
                + self.config.reward_w_sleep_excess * sleep_excess_penalty
                + self.config.reward_w_pue * pue_penalty
                + self.config.reward_w_sticky_config * sticky_config_penalty
                + self.config.reward_w_migration_count * migration_count_penalty
                + self.config.reward_w_guard_override * guard_changed_action
                + 0.30 * wake_penalty
                + invalid_action_penalty
            )
            + self.config.reward_w_util * combined_util_bonus
            + self.config.reward_w_off_bonus * off_bonus
            + power_off_bonus
        )

        if sla > self.config.sla_extra_threshold_1:
            reward -= self.config.sla_extra_penalty_1 * sla

        if sla > self.config.sla_extra_threshold_2:
            reward -= self.config.sla_extra_penalty_2 * sla

        info = {
            "dvfs_penalty": dvfs_penalty,
            "desired_dvfs": desired_dvfs,
            "dvfs_mismatch": dvfs_mismatch,
            "sticky_config_penalty": sticky_config_penalty,
            "same_config_steps": self.same_config_steps,
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
            "spare_ratio": spare_ratio,
            "overprovision_penalty": overprovision_penalty,
            "raw_action": int(raw_action),
            "executed_action": int(action),
            "guard_changed_action": int(guard_changed_action),
            "guard_override_penalty": float(self.config.reward_w_guard_override * guard_changed_action),
            "util_ratio": util_ratio,
            "latency_penalty": latency_penalty,
            "migration_plan": migration_events,
            "migration_events": migration_events,
            "migration_cost": migration_penalty,
            "migration_count_ratio": migration_count_ratio,
            "migration_count_penalty": migration_count_penalty,
            "hotspot_penalty": hotspot_penalty,
            "vm_to_host_map": vm_to_host_map,
            "vm_states": vm_states,
            "host_states": self._host_state_snapshot(host_utils),
            "uses_vm_snapshots": self.uses_vm_snapshots,
            "cooling_power": cooling_power,
            "host_loads": self.host_loads.copy().tolist(),
            "host_temps": self.host_temps.copy().tolist(),
            "host_status": self.host_status.copy().tolist(),
            "host_utils": host_utils.copy().tolist(),
            "desired_active_hosts": desired_active,
            "desired_sleep_hosts": desired_sleep,
            "active_excess_penalty": active_excess_penalty,
            "sleep_excess_penalty": sleep_excess_penalty,
            "off_bonus": off_bonus,
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

        obs = (
            self._observation()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
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
