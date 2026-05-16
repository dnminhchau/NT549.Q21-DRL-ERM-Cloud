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
    max_hosts: int = 8
    min_active_hosts: int = 1
    min_sleep_hosts: int = 0
    episode_length: int = 320
    # DVFS levels are normalized CPU frequency levels.
    # Default excludes 1.2 to avoid assuming overclocking in cloud servers.
    # If you need to reproduce older experiments, explicitly override
    # dvfs_levels=(0.6, 0.8, 1.0, 1.2) in the notebook.
    dvfs_levels: tuple[float, ...] = (0.6, 0.8, 1.0)
    host_nominal_capacity: float = 1.0 / 7.0

    p_idle: float = 80.0
    p_peak: float = 200.0
    p_sleep: float = 10.0
    p_off: float = 0.3
    host_switch_cost: float = 5.0

    # Dynamic CPU power follows P_dyn = C * V^2 * f.
    # A common simplified DVFS assumption is V proportional to f,
    # therefore P_dyn is approximately proportional to f^3.
    # Set to 2.0 only if you explicitly want the older quadratic model
    # for ablation/backward comparison.
    power_dvfs_exponent: float = 3.0

    migration_cost: float = 1.5

    priority_migration_multiplier: float = 3.0   
    memory_migration_scale: float = 0.01          
    migration_max_size_factor: float = 5.0        
    # Small tolerance to keep a VM on its previous host before forcing migration.
    # This reduces artificial migration churn in grouped VM snapshots.
    # Example: 0.02 means a VM can stay if it exceeds remaining capacity by <= 2% host capacity.
    migration_keep_tolerance: float = 0.02

    reward_w_energy: float = 2.60
    reward_w_sla: float = 4.50
    reward_w_switch: float = 0.08
    reward_w_migration: float = 0.12
    reward_w_latency: float = 0.35
    reward_w_util: float = 0.18
    reward_w_temp: float = 0.23
    reward_w_lifetime: float = 0.10
    # Optional facility-efficiency terms. Keep 0.0 by default for backward compatibility.
    # reward_w_pue penalizes PUE above pue_target; reward_w_cooling penalizes cooling overhead.
    # Use these only in ablation / PUE-temp experiments because PUE can conflict with total-energy minimization.
    reward_w_pue: float = 0.06
    reward_w_cooling: float = 0.10
    reward_w_overprovision: float = 1.25
    reward_w_active_excess: float = 1.60
    reward_w_sleep_excess: float = 0.30
    reward_w_off_bonus: float = 0.30

    reward_w_dvfs: float = 0.55
    reward_w_dvfs_mismatch: float = 0.45
    reward_w_sticky_config: float = 0.08
    # Configurable safe power-off bonus. Older code used a hard-coded +0.05.
    reward_w_power_off_action: float = 0.10
    power_off_bonus_sla_threshold: float = 0.025
    power_off_bonus_spare_threshold: float = 0.06
    # V5: do not reward deep power-off when load is clearly rising soon.
    power_off_bonus_demand_rise_threshold: float = 0.04
    # V5: small, bounded state bonus for keeping only the desired number of hosts OFF
    # in safe low-load windows. This borrows the good phase behavior from older runs
    # while reducing reward hacking: it is gated by SLA, spare capacity, demand trend,
    # and desired_off_hosts. Keep modest (<=0.08) in notebooks.
    reward_w_safe_off_state: float = 0.015
    # V5.1: make the low-load/off-host objective configurable instead of hard-coding 0.35.
    # This is intentionally separate from safe_off_state: off_shortage penalizes too few OFF hosts
    # in safe low-load windows, while safe_off_state gives a small bounded bonus for the desired OFF hosts.
    reward_w_off_shortage: float = 0.65

# V5.2: encourage the agent to shut down hosts when demand transitions
# from medium/high to low, but strongly discourage sleep/off actions in
    # clearly high-load states.
    reward_w_low_transition_action: float = 0.18
    low_transition_demand_threshold: float = 0.45
    low_transition_drop_threshold: float = 0.03
    reward_w_high_load_sleep_off: float = 0.40
    high_load_threshold: float = 0.75
    high_load_min_desired_active: int = 7
    mask_sleep_off_in_high_load: bool = True
    # V5.3: stronger low-load scale-down pressure without forcing action spam.
# These terms are gated by demand, so they do not encourage SLEEP/OFF in high phase.
    reward_w_low_active_excess: float = 0.95
    reward_w_low_scale_down_action: float = 0.12
    low_load_threshold: float = 0.55
    low_active_margin: int = 1

    # V5.3: fixed migration normalization for more stable reward scale.
# If None, the older dynamic scale based on current number of VM chunks is used.
    migration_norm_scale: float = 120.0

    # V5.3: use one smoother SLA penalty instead of allowing multiple cliffs to dominate.
    use_smooth_sla_penalty: bool = True
    sla_soft_threshold: float = 0.02
    sla_hard_threshold: float = 0.10
    sla_soft_weight: float = 3.0
    sla_hard_weight: float = 8.0
    # V6: thermal-aware placement and high-temperature DVFS guard.
    placement_thermal_weight: float = 0.030
    placement_hot_util_weight: float = 0.035
    placement_target_util_for_temp: float = 0.88
    reward_w_high_temp_under_dvfs: float = 0.30
    reward_w_hot_dvfs_up_action: float = 0.08
    high_temp_under_dvfs_threshold_c: float = 58.0

    # V8: macro-actions and conditional inaction penalty.
    # Goal: reduce policy conservatism / KEEP bias without returning to v5.1 action spam.
    reward_w_conditional_inaction: float = 0.20
    reward_w_macro_scale_down_action: float = 0.08
    reward_w_macro_power_off_action: float = 0.06
    inaction_demand_threshold: float = 0.60
    inaction_transition_drop_threshold: float = 0.025
    inaction_active_margin: int = 1
    macro_scale_down_demand_threshold: float = 0.75
    macro_power_off_demand_threshold: float = 0.55
    macro_active_margin: int = 1
    macro_max_switches_per_step: int = 1
    macro_block_if_demand_rising: bool = True
    macro_demand_rise_threshold: float = 0.035

    # V5.1: high-temperature/high-utilization guard. In high demand, lowering DVFS too much may
    # reduce energy but increase utilization and simulated temperature. This penalty discourages
    # the policy from sitting at high utilization when temperature is already high.
    reward_w_hot_util: float = 0.30
    hot_util_temp_threshold_c: float = 55.0
    hot_util_util_threshold: float = 0.88

    # Extra SLA guard used in stricter experiments. Keep 0.0 for backward compatibility.
    reward_w_sla_guard: float = 0.50
    sla_guard_threshold: float = 0.02

    # Reporting-only group weights used to explain the reward formulation
    # in the report. The actual reward remains the explicit weighted sum
    # below, but trace logs now aggregate components into these groups:
    # QoS/SLA, Energy, Resource efficiency, Operational overhead, Thermal/lifetime.
    reward_group_qos_weight_pct: float = 35.0
    reward_group_energy_weight_pct: float = 30.0
    reward_group_resource_weight_pct: float = 15.0
    reward_group_overhead_weight_pct: float = 10.0
    reward_group_thermal_weight_pct: float = 10.0

    target_host_util: float = 0.82
    reserve_sleep_hosts: int = 1

    base_pue: float = 1.18
    cooling_alpha: float = 0.08
    cooling_beta: float = 0.18
    cooling_fixed_power: float = 25.0
    # PUE/cooling reward normalization.
    pue_target: float = 1.35
    pue_cap: float = 2.50
    cooling_power_norm_cap: float = 500.0

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

    # V5: phase-balanced training/evaluation support.
    # If enabled, reset() samples episode start indices uniformly across low/medium/high
    # instead of sampling uniformly over the whole workload. This avoids learning only
    # from the longest phase and makes low/medium/high behavior measurable.
    balanced_phase_reset: bool = False
    phase_start_indices: dict[str, list[int]] | None = None
    phase_sampling_order: tuple[str, ...] = ("low", "medium", "high")
    phase_sampling_probs: tuple[float, ...] | None = None

    # V5: improve state coverage and safer power-off decisions.
    randomize_dvfs_init_by_demand: bool = True
    obs_include_demand_trend: bool = True
    obs_demand_trend_offset: int = 3


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
    ACTION_SCALE_DOWN_EXCESS = 7
    ACTION_POWER_OFF_EXCESS = 8

    ACTION_NAMES = {
        ACTION_KEEP: "KEEP",
        ACTION_WAKE_ONE: "WAKE_ONE",
        ACTION_SLEEP_ONE: "SLEEP_ONE",
        ACTION_DVFS_UP: "DVFS_UP",
        ACTION_DVFS_DOWN: "DVFS_DOWN",
        ACTION_POWER_OFF_ONE: "POWER_OFF_ONE",
        ACTION_BOOT_ONE: "BOOT_ONE",
        ACTION_SCALE_DOWN_EXCESS: "SCALE_DOWN_EXCESS",
        ACTION_POWER_OFF_EXCESS: "POWER_OFF_EXCESS",
    }

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

        self.action_space = spaces.Discrete(9)
        self.obs_dim = 19 if self.config.obs_include_demand_trend else 17
        self.observation_space = spaces.Box(
            # Most features are non-negative, but demand_trend can be negative.
            low=np.full(self.obs_dim, -self.config.obs_clip_high, dtype=np.float32),
            high=np.full(self.obs_dim, self.config.obs_clip_high, dtype=np.float32),
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
        self.last_reset_phase: str = "random"
        self.last_reset_start_idx: int = 0
        self.last_demand: float = 0.0

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

    def _choose_start_idx(self, options: dict[str, Any] | None = None) -> tuple[int, str]:
        """Choose the episode start index.

        V5 supports phase-balanced training and fixed evaluation windows:
        - options={"start_idx": i, "phase": "low"} forces a deterministic window.
        - config.balanced_phase_reset=True samples uniformly across provided phase starts.
        - fallback is the original uniform random start over the whole workload.
        """
        max_start = self.workload.size - self.config.episode_length - 1
        max_start = max(0, int(max_start))

        if options and "start_idx" in options:
            requested = int(options["start_idx"])
            requested = int(np.clip(requested, 0, max_start))
            return requested, str(options.get("phase", "fixed"))

        phase_starts = self.config.phase_start_indices or {}
        if options and "phase" in options and phase_starts:
            phase = str(options["phase"])
            starts = [int(s) for s in phase_starts.get(phase, []) if 0 <= int(s) <= max_start]
            if starts:
                return int(self.rng.choice(starts)), phase

        if self.config.balanced_phase_reset and phase_starts:
            phases = [p for p in self.config.phase_sampling_order if phase_starts.get(p)]
            phases = [p for p in phases if any(0 <= int(s) <= max_start for s in phase_starts.get(p, []))]
            if phases:
                probs = None
                if self.config.phase_sampling_probs is not None:
                    raw = np.asarray(self.config.phase_sampling_probs, dtype=np.float64)
                    if raw.size == len(self.config.phase_sampling_order):
                        prob_map = dict(zip(self.config.phase_sampling_order, raw))
                        probs = np.asarray([max(0.0, float(prob_map.get(p, 0.0))) for p in phases], dtype=np.float64)
                        if probs.sum() <= 0:
                            probs = None
                        else:
                            probs = probs / probs.sum()
                phase = str(self.rng.choice(phases, p=probs))
                starts = [int(s) for s in phase_starts[phase] if 0 <= int(s) <= max_start]
                return int(self.rng.choice(starts)), phase

        return int(self.rng.integers(0, max(1, max_start + 1))), "random"

    def _initial_dvfs_idx(self, demand: float) -> int:
        """Demand-aware randomized DVFS initialization for better state coverage."""
        n = len(self.config.dvfs_levels)
        if n <= 1:
            return 0
        if not self.config.randomize_dvfs_init_by_demand:
            return min(2, n - 1)
        if demand < 0.35:
            high = min(1, n - 1)
            return int(self.rng.integers(0, high + 1))
        if demand < 0.70:
            low = min(1, n - 1)
            high = min(2, n - 1)
            return int(self.rng.integers(low, high + 1))
        return min(2, n - 1)

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

        elif action == self.ACTION_SCALE_DOWN_EXCESS:
            # V8.1 macro-action: move a bounded number of excess active hosts to Sleep.
            # Block when demand is rising soon to avoid Sleep/Wake oscillation and migration churn.
            demand_now = self._get_demand(0)
            demand_next = self._get_demand(1)
            demand_rising_for_macro = bool(
                self.config.macro_block_if_demand_rising
                and demand_next > demand_now + float(self.config.macro_demand_rise_threshold)
            )
            if demand_now < self.config.macro_scale_down_demand_threshold and not demand_rising_for_macro:
                desired_active = self._desired_active_hosts(demand_now)
                max_switches = max(1, int(self.config.macro_max_switches_per_step))
                while (
                    self.active_hosts > max(self.config.min_active_hosts, desired_active + self.config.macro_active_margin)
                    and switches < max_switches
                ):
                    active_idx = self._active_indices()
                    if active_idx.size <= self.config.min_active_hosts:
                        break
                    target = int(active_idx[-1])
                    self.host_status[target] = self.STATUS_SLEEP
                    self.host_loads[target] = 0.0
                    switches += 1

        elif action == self.ACTION_POWER_OFF_EXCESS:
            # V8.1 macro-action: move a bounded number of excess Sleep hosts to Off.
            # More conservative than v8: do not power off if demand is rising soon.
            demand_now = self._get_demand(0)
            demand_next = self._get_demand(1)
            demand_rising_for_macro = bool(
                self.config.macro_block_if_demand_rising
                and demand_next > demand_now + float(self.config.macro_demand_rise_threshold)
            )
            if demand_now < self.config.macro_power_off_demand_threshold and not demand_rising_for_macro:
                desired_active = self._desired_active_hosts(demand_now)
                desired_sleep = self._desired_sleep_hosts(demand_now, demand_next)
                desired_off = max(0, self.config.max_hosts - desired_active - desired_sleep)
                max_switches = max(1, int(self.config.macro_max_switches_per_step))
                while (
                    self.sleep_hosts > max(self.config.min_sleep_hosts, desired_sleep)
                    and self.off_hosts < desired_off
                    and switches < max_switches
                ):
                    sleep_idx = self._sleep_indices()
                    if sleep_idx.size <= self.config.min_sleep_hosts:
                        break
                    target = int(sleep_idx[-1])
                    self.host_status[target] = self.STATUS_OFF
                    self.host_loads[target] = 0.0
                    switches += 1

        return switches, wake_from_off

    def valid_action_mask(self) -> np.ndarray:
        """Return True for actions that are valid in the current state.

        This method is intentionally lightweight so it can be used by
        sb3-contrib MaskablePPO. It prevents no-op / invalid actions such as
        BOOT_ONE when there is no OFF host or DVFS_UP at the maximum DVFS level.
        V5.2 also masks SLEEP_ONE / POWER_OFF_ONE in clearly high-load states
        where the desired active host count is already near the maximum.
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        if self.sleep_hosts == 0:
            mask[self.ACTION_WAKE_ONE] = False
        if self.active_hosts <= self.config.min_active_hosts:
            mask[self.ACTION_SLEEP_ONE] = False
        if self.dvfs_idx >= len(self.config.dvfs_levels) - 1:
            mask[self.ACTION_DVFS_UP] = False
        if self.dvfs_idx <= 0:
            mask[self.ACTION_DVFS_DOWN] = False
        if self.sleep_hosts <= self.config.min_sleep_hosts:
            mask[self.ACTION_POWER_OFF_ONE] = False
        if self.off_hosts == 0:
            mask[self.ACTION_BOOT_ONE] = False

        demand_now = self._get_demand(0)
        demand_next = self._get_demand(1)
        demand_rising_for_macro = bool(
            self.config.macro_block_if_demand_rising
            and demand_next > demand_now + float(self.config.macro_demand_rise_threshold)
        )
        desired_active = self._desired_active_hosts(demand_now)
        desired_sleep = self._desired_sleep_hosts(demand_now, demand_next)
        desired_off = max(0, self.config.max_hosts - desired_active - desired_sleep)

        if (
            demand_now >= self.config.macro_scale_down_demand_threshold
            or demand_rising_for_macro
            or self.active_hosts <= desired_active + self.config.macro_active_margin
        ):
            mask[self.ACTION_SCALE_DOWN_EXCESS] = False

        if (
            demand_now >= self.config.macro_power_off_demand_threshold
            or demand_rising_for_macro
            or self.sleep_hosts <= max(self.config.min_sleep_hosts, desired_sleep)
            or self.off_hosts >= desired_off
        ):
            mask[self.ACTION_POWER_OFF_EXCESS] = False

        if self.config.mask_sleep_off_in_high_load:
            if demand_now >= self.config.high_load_threshold and desired_active >= self.config.high_load_min_desired_active:
                mask[self.ACTION_SLEEP_ONE] = False
                mask[self.ACTION_POWER_OFF_ONE] = False
                mask[self.ACTION_SCALE_DOWN_EXCESS] = False
                mask[self.ACTION_POWER_OFF_EXCESS] = False
        return mask

    def action_masks(self) -> np.ndarray:
        """Compatibility alias required by sb3-contrib MaskablePPO."""
        return self.valid_action_mask()

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
                # Keep VM on its previous host if possible. A small configurable
                # tolerance can be enabled in ablation runs to reduce migration
                # churn from grouped VM snapshots whose chunk sizes change over time.
                keep_tolerance = max(0.0, float(self.config.migration_keep_tolerance)) * host_capacity
                if remaining[previous_host] + keep_tolerance >= demand_unit - 1e-9:
                    chosen_host = int(previous_host)

            # 2) Best-fit: chọn host còn đủ chỗ và dư ít nhất sau khi đặt.
            if chosen_host is None:
                feasible_hosts = [h for h, rem in remaining.items() if rem >= demand_unit - 1e-9]
                if feasible_hosts:
                    def _placement_score(h: int) -> float:
                        rem_after = remaining[h] - demand_unit
                        util_after = (host_capacity - rem_after) / max(host_capacity, 1e-8)
                        temp_norm = max(
                            0.0,
                            (float(self.host_temps[h]) - self.config.reference_temp_c)
                            / max(self.config.max_safe_temp_c - self.config.reference_temp_c, 1e-8),
                        )
                        hot_util_excess = max(
                            0.0,
                            util_after - float(self.config.placement_target_util_for_temp),
                        )
                        return (
                            rem_after
                            + float(self.config.placement_thermal_weight) * temp_norm
                            + float(self.config.placement_hot_util_weight) * hot_util_excess
                        )

                    chosen_host = min(feasible_hosts, key=_placement_score)
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

            # Power model: P_host = P_idle + P_dynamic.
            # Dynamic power is modeled as utilization * f^alpha.
            # With CMOS DVFS, P_dyn = C * V^2 * f; assuming V roughly
            # scales with f gives alpha≈3. This is still a simulation
            # proxy, not hardware calibration.
            dvfs_power_scale = self.dvfs ** float(self.config.power_dvfs_exponent)
            active_power = (
                self.config.p_idle
                + (self.config.p_peak - self.config.p_idle)
                * np.minimum(utils, 1.0)
                * dvfs_power_scale
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

        features = [
            demand_now,
            demand_next,
            active_ratio,
            self.dvfs / max(self.config.dvfs_levels),
            self.last_power_total / (
                self.config.max_hosts
                * self.config.p_peak
                * max(self.config.dvfs_levels) ** float(self.config.power_dvfs_exponent)
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
        ]

        if self.config.obs_include_demand_trend:
            trend_offset = max(1, int(self.config.obs_demand_trend_offset))
            demand_future = self._get_demand(trend_offset)
            demand_trend = float(np.clip((demand_future - demand_now) / max(demand_now, 0.01), -1.0, 1.0))
            demand_rolling_mean = float(np.mean([self._get_demand(i) for i in range(trend_offset + 1)]))
            features.extend([demand_trend, demand_rolling_mean])

        obs = np.asarray(features, dtype=np.float32)
        return np.clip(obs, -self.config.obs_clip_high, self.config.obs_clip_high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.start_idx, self.last_reset_phase = self._choose_start_idx(options)
        self.last_reset_start_idx = int(self.start_idx)
        self.step_idx = 0

        demand = self._get_demand(0)
        self.dvfs_idx = self._initial_dvfs_idx(demand)
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
        self.last_demand = float(demand)

        return self._observation(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Action không hợp lệ: {action}"

        action = int(action)
        valid_action_mask = self.valid_action_mask()
        action_valid_before_step = bool(valid_action_mask[action])
        demand_prev = float(self.last_demand)

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
        elif action == self.ACTION_SCALE_DOWN_EXCESS:
            demand_now = self._get_demand(0)
            demand_next_now = self._get_demand(1)
            demand_rising_for_macro = bool(
                self.config.macro_block_if_demand_rising
                and demand_next_now > demand_now + float(self.config.macro_demand_rise_threshold)
            )
            desired_active_now = self._desired_active_hosts(demand_now)
            if (
                demand_now >= self.config.macro_scale_down_demand_threshold
                or demand_rising_for_macro
                or self.active_hosts <= desired_active_now + self.config.macro_active_margin
            ):
                invalid_action_penalty = 0.05
        elif action == self.ACTION_POWER_OFF_EXCESS:
            demand_now = self._get_demand(0)
            demand_next_now = self._get_demand(1)
            demand_rising_for_macro = bool(
                self.config.macro_block_if_demand_rising
                and demand_next_now > demand_now + float(self.config.macro_demand_rise_threshold)
            )
            desired_active_now = self._desired_active_hosts(demand_now)
            desired_sleep_now = self._desired_sleep_hosts(demand_now, demand_next_now)
            desired_off_now = max(0, self.config.max_hosts - desired_active_now - desired_sleep_now)
            if (
                demand_now >= self.config.macro_power_off_demand_threshold
                or demand_rising_for_macro
                or self.sleep_hosts <= max(self.config.min_sleep_hosts, desired_sleep_now)
                or self.off_hosts >= desired_off_now
            ):
                invalid_action_penalty = 0.05

        switches, wake_from_off = self._apply_action(action)

        demand = self._get_demand(0)
        demand_drop = max(0.0, demand_prev - demand)

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

        sticky_config_penalty_raw = min(self.same_config_steps / 20.0, 1.0)

        self.prev_active_hosts = int(self.active_hosts)
        self.prev_dvfs = float(self.dvfs)

        # Normalize by the same DVFS exponent used in the power model.
        # Earlier versions used **2 here while _compute_power() used f^power_dvfs_exponent;
        # this caused a small inconsistency in reward scaling.
        normalized_energy = power_total / (
            self.config.max_hosts
            * self.config.p_peak
            * max(self.config.dvfs_levels) ** float(self.config.power_dvfs_exponent)
            + self.config.cooling_fixed_power
        )

        utilization_bonus = min(mean_util, 1.0)
        balanced_util_bonus = max(0.0, 1.0 - abs(mean_util - 0.78) / 0.78)

        avg_temp = float(np.mean(self.host_temps))
        thermal_penalty = max(0.0, avg_temp - self.config.reference_temp_c) / max(
            self.config.max_safe_temp_c - self.config.reference_temp_c, 1.0
        )
        # PUE penalty is only the excess above a target. This avoids rewarding artificially
        # high IT power simply because the PUE ratio can decrease when IT power rises.
        pue_penalty = np.clip(
            max(0.0, pue - self.config.pue_target)
            / max(self.config.pue_cap - self.config.pue_target, 1e-8),
            0.0,
            1.0,
        )
        # Cooling overhead penalty uses absolute cooling power, so it is safer than directly
        # minimizing raw PUE alone.
        cooling_penalty = np.clip(
            cooling_power / max(self.config.cooling_power_norm_cap, 1e-8),
            0.0,
            1.0,
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
        desired_off = max(0, self.config.max_hosts - desired_active - desired_sleep)
        demand_rising_soon = bool(
            demand_next > demand + float(self.config.power_off_bonus_demand_rise_threshold)
        )

        active_excess_penalty = max(0.0, self.active_hosts - desired_active) / max(
            self.config.max_hosts, 1
        )
        sleep_excess_penalty = max(0.0, self.sleep_hosts - desired_sleep) / max(
            self.config.max_hosts, 1
        )
        low_or_medium_window = bool(demand < self.config.high_load_threshold)
        off_shortage_penalty = (
            max(0.0, desired_off - self.off_hosts) / max(self.config.max_hosts, 1)
            if low_or_medium_window and desired_off > 0
            else 0.0
        )

        in_high_load_guard = bool(
            demand >= self.config.high_load_threshold
            and desired_active >= self.config.high_load_min_desired_active
        )
        high_load_sleep_off_penalty = 1.0 if (
            in_high_load_guard
            and action in (
                self.ACTION_SLEEP_ONE,
                self.ACTION_POWER_OFF_ONE,
                self.ACTION_SCALE_DOWN_EXCESS,
                self.ACTION_POWER_OFF_EXCESS,
            )
        ) else 0.0

        low_transition_action_bonus = 0.0
        if (
            demand <= self.config.low_transition_demand_threshold
            and demand_drop >= self.config.low_transition_drop_threshold
            and self.active_hosts > desired_active
            and action_valid_before_step
            and switches > 0
            and action in (self.ACTION_SLEEP_ONE, self.ACTION_POWER_OFF_ONE)
        ):
            low_transition_action_bonus = 1.0

        # V5.3: if the workload is already low/medium-low and active hosts are still
        # above the desired level, KEEP should be penalized through conditional sticky
        # and SLEEP/OFF should receive a small action bonus.
        low_active_excess_penalty = 0.0
        if demand < self.config.low_load_threshold:
            low_active_excess_penalty = max(
                0.0,
                self.active_hosts - desired_active - self.config.low_active_margin,
            ) / max(self.config.max_hosts, 1)

        low_scale_down_action_bonus = 0.0
        if (
            demand < self.config.low_load_threshold
            and self.active_hosts > desired_active + self.config.low_active_margin
            and action_valid_before_step
            and switches > 0
            and action in (self.ACTION_SLEEP_ONE, self.ACTION_POWER_OFF_ONE)
        ):
            low_scale_down_action_bonus = 1.0

        macro_scale_down_action_bonus = 0.0
        if (
            action == self.ACTION_SCALE_DOWN_EXCESS
            and action_valid_before_step
            and switches > 0
            and demand < self.config.macro_scale_down_demand_threshold
        ):
            macro_scale_down_action_bonus = min(1.0, switches / max(1, self.config.macro_max_switches_per_step))

        macro_power_off_action_bonus = 0.0
        if (
            action == self.ACTION_POWER_OFF_EXCESS
            and action_valid_before_step
            and switches > 0
            and demand < self.config.macro_power_off_demand_threshold
        ):
            macro_power_off_action_bonus = min(1.0, switches / max(1, self.config.macro_max_switches_per_step))

        conditional_inaction_penalty = 0.0
        active_surplus = max(0.0, self.active_hosts - desired_active - self.config.inaction_active_margin)
        if (
            action == self.ACTION_KEEP
            and active_surplus > 0
            and demand < self.config.inaction_demand_threshold
            and sla <= self.config.sla_guard_threshold
        ):
            conditional_inaction_penalty = active_surplus / max(self.config.max_hosts, 1)

        if (
            action == self.ACTION_KEEP
            and active_surplus > 0
            and demand_drop >= self.config.inaction_transition_drop_threshold
            and demand < self.config.high_load_threshold
            and sla <= self.config.sla_guard_threshold
        ):
            conditional_inaction_penalty = max(
                conditional_inaction_penalty,
                1.15 * active_surplus / max(self.config.max_hosts, 1),
            )

        # V5.1 thermal guard: penalize high utilization only when the simulated temperature
        # is already above a threshold. This avoids blindly increasing DVFS everywhere,
        # but nudges high-demand/high-temperature states away from the hottest operating point.
        hot_temp_ratio = max(0.0, avg_temp - self.config.hot_util_temp_threshold_c) / max(
            self.config.max_safe_temp_c - self.config.hot_util_temp_threshold_c, 1.0
        )
        hot_util_ratio = max(0.0, mean_util - self.config.hot_util_util_threshold) / max(
            1.5 - self.config.hot_util_util_threshold, 1e-8
        )
        hot_util_penalty = float(np.clip(hot_temp_ratio * hot_util_ratio, 0.0, 1.0))

        max_dvfs = max(self.config.dvfs_levels)
        min_dvfs = min(self.config.dvfs_levels)
        high_temp_under_dvfs_penalty = 0.0
        hot_dvfs_up_action_bonus = 0.0
        if (
            demand >= self.config.high_load_threshold
            and avg_temp >= self.config.high_temp_under_dvfs_threshold_c
            and self.dvfs < max_dvfs - 1e-9
        ):
            high_temp_under_dvfs_penalty = (max_dvfs - self.dvfs) / max(max_dvfs - min_dvfs, 1e-8)
            if action_valid_before_step and action == self.ACTION_DVFS_UP:
                hot_dvfs_up_action_bonus = 1.0

        wake_penalty = 0.0
        if action in (self.ACTION_WAKE_ONE, self.ACTION_BOOT_ONE) and self.active_hosts > desired_active:
            wake_penalty = max(spare_ratio, active_excess_penalty)

        # V5 safe-off reward design:
        # - Avoid the old reward-hacking issue where the agent could get a bonus just
        #   for keeping many hosts OFF forever.
        # - Still encourage useful OFF-host behavior in low-load phases, but only when
        #   the number of OFF hosts is bounded by desired_off and the next demand is not rising.
        safe_off_state_bonus = 0.0
        if (
            self.config.reward_w_safe_off_state > 0.0
            and sla < self.config.power_off_bonus_sla_threshold
            and spare_ratio > self.config.power_off_bonus_spare_threshold
            and not demand_rising_soon
            and desired_off > 0
            and low_or_medium_window
        ):
            safe_off_state_bonus = float(min(self.off_hosts, desired_off) / max(self.config.max_hosts, 1))

        # Backward-compatible name kept in logs. It is now the bounded safe-off state bonus.
        off_bonus = safe_off_state_bonus

        # Direct action bonus: reward the agent only when it actually performs a valid
        # POWER_OFF_ONE action in a safe low/rising-load condition.
        power_off_bonus = 0.0
        if (
            action_valid_before_step
            and action == self.ACTION_POWER_OFF_ONE
            and switches > 0
            and sla < self.config.power_off_bonus_sla_threshold
            and spare_ratio > self.config.power_off_bonus_spare_threshold
            and not demand_rising_soon
            and low_or_medium_window
        ):
            power_off_bonus = float(self.config.reward_w_power_off_action)

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
        dynamic_migration_scale = max_penalty_per_vm * max(len(vm_states), 1)
        migration_scale = (
            float(self.config.migration_norm_scale)
            if self.config.migration_norm_scale is not None and self.config.migration_norm_scale > 0
            else dynamic_migration_scale
        )
        self.last_migration_penalty_weighted = float(
            np.clip(migration_penalty / max(migration_scale, 1e-8), 0.0, self.config.obs_clip_high)
        )

        if self.config.use_smooth_sla_penalty:
            sla_penalty = (
                sla
                + self.config.sla_soft_weight * max(0.0, sla - self.config.sla_soft_threshold) ** 2
                + self.config.sla_hard_weight * max(0.0, sla - self.config.sla_hard_threshold) ** 2
            )
        else:
            sla_penalty = sla * (
                1.0
                + self.config.sla_penalty_growth_factor
                * max(0.0, sla - self.config.sla_penalty_growth_threshold)
            )
        sla_guard_penalty = max(0.0, sla - self.config.sla_guard_threshold)
        latency_clipped = min(latency_penalty / self.config.latency_cap, 1.0)

        config_waste = (
            active_excess_penalty
            + overprovision_penalty
            + off_shortage_penalty
            + low_active_excess_penalty
            + 0.5 * max(0.0, thermal_penalty - 0.20)
        )
        sticky_config_penalty = sticky_config_penalty_raw * float(np.clip(config_waste, 0.0, 1.0))

        combined_util_bonus = (
            0.5 * min(mean_util, 1.0)
            + 0.5 * max(0.0, 1.0 - abs(mean_util - 0.78) / 0.78)
        )

        migration_penalty_norm = self.last_migration_penalty_weighted
        lifetime_penalty_norm = lifetime_penalty / max(self.config.max_hosts, 1)

        reward = (
            -(
                self.config.reward_w_energy * normalized_energy
                + self.config.reward_w_sla * sla_penalty
                + self.config.reward_w_sla_guard * sla_guard_penalty
                + self.config.reward_w_latency * latency_clipped
                + self.config.reward_w_switch * (switches / max(self.config.max_hosts, 1))
                + self.config.reward_w_migration * migration_penalty_norm
                + self.config.reward_w_temp * thermal_penalty
                + self.config.reward_w_lifetime * lifetime_penalty_norm
                + self.config.reward_w_pue * pue_penalty
                + self.config.reward_w_cooling * cooling_penalty
                + self.config.reward_w_hot_util * hot_util_penalty
                + self.config.reward_w_high_temp_under_dvfs * high_temp_under_dvfs_penalty
                + self.config.reward_w_overprovision * overprovision_penalty
                + self.config.reward_w_dvfs * dvfs_penalty
                + self.config.reward_w_dvfs_mismatch * dvfs_mismatch
                + self.config.reward_w_active_excess * active_excess_penalty
                + self.config.reward_w_sleep_excess * sleep_excess_penalty
                + self.config.reward_w_off_shortage * off_shortage_penalty
                + self.config.reward_w_low_active_excess * low_active_excess_penalty
                + self.config.reward_w_conditional_inaction * conditional_inaction_penalty
                + self.config.reward_w_high_load_sleep_off * high_load_sleep_off_penalty
                + self.config.reward_w_sticky_config * sticky_config_penalty
                + 0.30 * wake_penalty
                + invalid_action_penalty
            )
            + self.config.reward_w_util * combined_util_bonus
            + self.config.reward_w_off_bonus * off_bonus
            + self.config.reward_w_low_transition_action * low_transition_action_bonus
            + self.config.reward_w_low_scale_down_action * low_scale_down_action_bonus
            + self.config.reward_w_macro_scale_down_action * macro_scale_down_action_bonus
            + self.config.reward_w_macro_power_off_action * macro_power_off_action_bonus
            + self.config.reward_w_hot_dvfs_up_action * hot_dvfs_up_action_bonus
            + power_off_bonus
        )

        if not self.config.use_smooth_sla_penalty:
            if sla > self.config.sla_extra_threshold_1:
                reward -= self.config.sla_extra_penalty_1 * sla

            if sla > self.config.sla_extra_threshold_2:
                reward -= self.config.sla_extra_penalty_2 * sla

        reward_components = {
            "energy_penalty": float(normalized_energy),
            "sla_penalty": float(sla_penalty),
            "sla_guard_penalty": float(sla_guard_penalty),
            "latency_penalty": float(latency_clipped),
            "switch_penalty": float(switches / max(self.config.max_hosts, 1)),
            "migration_penalty": float(migration_penalty_norm),
            "thermal_penalty": float(thermal_penalty),
            "lifetime_penalty": float(lifetime_penalty_norm),
            "pue_penalty": float(pue_penalty),
            "cooling_penalty": float(cooling_penalty),
            "hot_util_penalty": float(hot_util_penalty),
            "high_temp_under_dvfs_penalty": float(high_temp_under_dvfs_penalty),
            "hot_dvfs_up_action_bonus": float(hot_dvfs_up_action_bonus),
            "overprovision_penalty": float(overprovision_penalty),
            "dvfs_penalty": float(dvfs_penalty),
            "dvfs_mismatch": float(dvfs_mismatch),
            "active_excess_penalty": float(active_excess_penalty),
            "sleep_excess_penalty": float(sleep_excess_penalty),
            "off_shortage_penalty": float(off_shortage_penalty),
            "low_active_excess_penalty": float(low_active_excess_penalty),
            "conditional_inaction_penalty": float(conditional_inaction_penalty),
            "active_surplus": float(active_surplus),
            "low_scale_down_action_bonus": float(low_scale_down_action_bonus),
            "macro_scale_down_action_bonus": float(macro_scale_down_action_bonus),
            "macro_power_off_action_bonus": float(macro_power_off_action_bonus),
            "high_load_sleep_off_penalty": float(high_load_sleep_off_penalty),
            "low_transition_action_bonus": float(low_transition_action_bonus),
            "safe_off_state_bonus": float(safe_off_state_bonus),
            "demand_rising_soon": float(demand_rising_soon),
            "sticky_config_penalty": float(sticky_config_penalty),
            "wake_penalty": float(wake_penalty),
            "invalid_action_penalty": float(invalid_action_penalty),
            "util_bonus": float(combined_util_bonus),
            "off_bonus": float(off_bonus),
            "power_off_bonus": float(power_off_bonus),
        }

        # Weighted scalarization: PPO optimizes a single scalar reward,
        # so multi-objective criteria are converted into a weighted sum.
        # These logs make the reward explainable during report/review.
        reward_weighted_components = {
            "energy": float(self.config.reward_w_energy * normalized_energy),
            "sla": float(self.config.reward_w_sla * sla_penalty),
            "sla_guard": float(self.config.reward_w_sla_guard * sla_guard_penalty),
            "latency": float(self.config.reward_w_latency * latency_clipped),
            "switch": float(self.config.reward_w_switch * (switches / max(self.config.max_hosts, 1))),
            "migration": float(self.config.reward_w_migration * migration_penalty_norm),
            "thermal": float(self.config.reward_w_temp * thermal_penalty),
            "lifetime": float(self.config.reward_w_lifetime * lifetime_penalty_norm),
            "pue": float(self.config.reward_w_pue * pue_penalty),
            "cooling": float(self.config.reward_w_cooling * cooling_penalty),
            "overprovision": float(self.config.reward_w_overprovision * overprovision_penalty),
            "dvfs": float(self.config.reward_w_dvfs * dvfs_penalty),
            "dvfs_mismatch": float(self.config.reward_w_dvfs_mismatch * dvfs_mismatch),
            "active_excess": float(self.config.reward_w_active_excess * active_excess_penalty),
            "sleep_excess": float(self.config.reward_w_sleep_excess * sleep_excess_penalty),
            "off_shortage": float(self.config.reward_w_off_shortage * off_shortage_penalty),
            "low_active_excess": float(self.config.reward_w_low_active_excess * low_active_excess_penalty),
            "conditional_inaction": float(self.config.reward_w_conditional_inaction * conditional_inaction_penalty),
            "high_load_sleep_off": float(self.config.reward_w_high_load_sleep_off * high_load_sleep_off_penalty),
            "sticky_config": float(self.config.reward_w_sticky_config * sticky_config_penalty),
            "wake": float(0.30 * wake_penalty),
            "invalid_action": float(invalid_action_penalty),
            "util_bonus": float(self.config.reward_w_util * combined_util_bonus),
            "low_transition_action_bonus": float(self.config.reward_w_low_transition_action * low_transition_action_bonus),
            "low_scale_down_action_bonus": float(self.config.reward_w_low_scale_down_action * low_scale_down_action_bonus),
            "macro_scale_down_action_bonus": float(self.config.reward_w_macro_scale_down_action * macro_scale_down_action_bonus),
            "macro_power_off_action_bonus": float(self.config.reward_w_macro_power_off_action * macro_power_off_action_bonus),
            "hot_dvfs_up_action_bonus": float(self.config.reward_w_hot_dvfs_up_action * hot_dvfs_up_action_bonus),
            "power_off_bonus": float(power_off_bonus),
        }
        reward_groups = {
            "qos_sla_latency_cost": float(
                reward_weighted_components["sla"]
                + reward_weighted_components["sla_guard"]
                + reward_weighted_components["latency"]
            ),
            "energy_cost": float(
                reward_weighted_components["energy"]
                + reward_weighted_components["dvfs"]
                + reward_weighted_components["dvfs_mismatch"]
                + reward_weighted_components["overprovision"]
                + reward_weighted_components["cooling"]
            ),
            "facility_efficiency_cost": float(
                reward_weighted_components["pue"]
                + reward_weighted_components["cooling"]
            ),
            "resource_efficiency_cost": float(
                reward_weighted_components["active_excess"]
                + reward_weighted_components["sleep_excess"]
                + reward_weighted_components["off_shortage"]
                + reward_weighted_components["low_active_excess"]
                + reward_weighted_components["conditional_inaction"]
                + reward_weighted_components["high_load_sleep_off"]
                + reward_weighted_components["sticky_config"]
                - reward_weighted_components["util_bonus"]
                - reward_weighted_components["low_transition_action_bonus"]
                - reward_weighted_components["low_scale_down_action_bonus"]
                - reward_weighted_components["macro_scale_down_action_bonus"]
                - reward_weighted_components["macro_power_off_action_bonus"]
            ),
            "operational_overhead_cost": float(
                reward_weighted_components["switch"]
                + reward_weighted_components["migration"]
                + reward_weighted_components["wake"]
                + reward_weighted_components["invalid_action"]
            ),
            "thermal_lifetime_cost": float(
                reward_weighted_components["thermal"]
                + reward_weighted_components["lifetime"]
                + reward_weighted_components["pue"]
            ),
            "safe_action_bonus": float(reward_weighted_components["power_off_bonus"] + reward_weighted_components["low_transition_action_bonus"] + reward_weighted_components["low_scale_down_action_bonus"] + reward_weighted_components["macro_scale_down_action_bonus"] + reward_weighted_components["macro_power_off_action_bonus"] + reward_weighted_components["hot_dvfs_up_action_bonus"]),
        }

        info = {
            "global_timestep": int(self.start_idx + self.step_idx),
            "reset_start_idx": int(self.last_reset_start_idx),
            "reset_phase": str(self.last_reset_phase),
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
            "pue_penalty": float(pue_penalty),
            "cooling_penalty": float(cooling_penalty),
            "lifetime_penalty": float(lifetime_penalty),
            "mean_host_age": float(np.mean(self.host_age)),
            "sla": sla,
            "sla_guard_penalty": float(sla_guard_penalty),
            "spare_ratio": spare_ratio,
            "overprovision_penalty": overprovision_penalty,
            "util_ratio": util_ratio,
            "latency_penalty": latency_penalty,
            "migration_plan": migration_events,
            "migration_events": migration_events,
            "migration_cost": migration_penalty,
            "vm_to_host_map": vm_to_host_map,
            "vm_states": vm_states,
            "host_states": self._host_state_snapshot(host_utils),
            "uses_vm_snapshots": self.uses_vm_snapshots,
            "cooling_power": cooling_power,
            "power_dvfs_exponent": float(self.config.power_dvfs_exponent),
            "host_loads": self.host_loads.copy().tolist(),
            "host_temps": self.host_temps.copy().tolist(),
            "host_status": self.host_status.copy().tolist(),
            "host_utils": host_utils.copy().tolist(),
            "desired_active_hosts": desired_active,
            "desired_sleep_hosts": desired_sleep,
            "desired_off_hosts": desired_off,
            "demand_rising_soon": demand_rising_soon,
            "active_excess_penalty": active_excess_penalty,
            "sleep_excess_penalty": sleep_excess_penalty,
            "off_shortage_penalty": off_shortage_penalty,
            "low_active_excess_penalty": low_active_excess_penalty,
            "conditional_inaction_penalty": conditional_inaction_penalty,
            "active_surplus": active_surplus,
            "low_scale_down_action_bonus": low_scale_down_action_bonus,
            "macro_scale_down_action_bonus": macro_scale_down_action_bonus,
            "macro_power_off_action_bonus": macro_power_off_action_bonus,
            "sticky_config_penalty_raw": sticky_config_penalty_raw,
            "config_waste": config_waste,
            "migration_norm_scale": migration_scale,
            "high_load_sleep_off_penalty": high_load_sleep_off_penalty,
            "low_transition_action_bonus": low_transition_action_bonus,
            "safe_off_state_bonus": safe_off_state_bonus,
            "off_bonus": off_bonus,
            "action": int(action),
            "action_name": self.ACTION_NAMES.get(int(action), f"UNKNOWN_{action}"),
            "action_valid": bool(action_valid_before_step),
            "valid_action_mask": valid_action_mask.astype(int).tolist(),
            "reward_total": float(reward),
            "reward_components": reward_components,
            "reward_weighted_components": reward_weighted_components,
            "reward_groups": reward_groups,
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
        self.last_demand = float(demand)

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
