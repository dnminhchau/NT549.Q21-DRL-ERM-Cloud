from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EpisodeMetrics:
    total_reward: float
    total_energy: float
    total_it_energy: float
    avg_power: float
    avg_pue: float
    sla_rate: float
    avg_active_hosts: float
    avg_sleep_hosts: float
    avg_off_hosts: float
    avg_temp: float
    total_switches: int
    total_migrations: int
    total_migration_cost: float = 0.0


class FixedPolicy:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs: np.ndarray):
        return self.action, None


class RoundRobinPolicy:

    def __init__(self):
        self.cursor = 0

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        demand_next = float(obs[1])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        sla_ratio = float(obs[5])
        sleep_ratio = float(obs[8])
        off_ratio = float(obs[9])

        self.cursor = (self.cursor + 1) % 4

        rising = demand_next > demand_now + 0.05
        high_load = demand_now > 0.85 or rising
        very_high_load = demand_now > 0.95 or sla_ratio > 0.03

        # Khi tải rất cao hoặc SLA xấu thì tăng tài nguyên.
        if very_high_load:
            if sleep_ratio > 0.0:
                return 1, None  # WAKE_ONE
            if off_ratio > 0.0 and self.cursor % 2 == 0:
                return 6, None  # BOOT_ONE
            if dvfs_ratio < 0.98:
                return 3, None  # DVFS_UP
            return 0, None

        # Khi tải tăng thì wake/boot nhẹ.
        if high_load and active_ratio < 0.95:
            if sleep_ratio > 0.0:
                return 1, None  # WAKE_ONE
            if off_ratio > 0.0 and self.cursor % 2 == 0:
                return 6, None  # BOOT_ONE
            return 0, None

        # RoundRobin basic không nên quá chủ động power-off.
        # Chỉ sleep khi tải rất thấp.
        if demand_now < 0.25 and active_ratio > 0.35:
            return 2, None  # SLEEP_ONE

        # DVFS_DOWN chỉ khi tải rất thấp.
        if demand_now < 0.30 and dvfs_ratio > 0.60:
            return 4, None  # DVFS_DOWN

        return 0, None  # KEEP


class ThresholdPolicy:
    
    def __init__(self, high: float = 0.85, low: float = 0.25):
        self.high = high
        self.low = low

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        demand_next = float(obs[1])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        sla_ratio = float(obs[5])
        sleep_ratio = float(obs[8])
        off_ratio = float(obs[9])

        if demand_now > self.high or demand_next > self.high or sla_ratio > 0.02:
            if active_ratio < 0.95:
                if sleep_ratio > 0.0:
                    return 1, None  # WAKE_ONE
                if off_ratio > 0.0:
                    return 6, None  # BOOT_ONE
                return 1, None  # fallback WAKE_ONE
            if dvfs_ratio < 0.98 and sla_ratio > 0.01:
                return 3, None  # DVFS_UP

        if demand_now < self.low:
            if active_ratio > 0.35:
                return 2, None  # SLEEP_ONE
            if dvfs_ratio > 0.60:
                return 4, None  # DVFS_DOWN

        return 0, None  # KEEP


class BestFitPolicy:

    def __init__(self, target_util: float = 0.70):
        self.target_util = target_util

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        demand_next = float(obs[1])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        sla_ratio = float(obs[5])
        sleep_ratio = float(obs[8])
        off_ratio = float(obs[9])

        # BestFit basic: ước lượng số host cần dựa trên demand.
        effective_demand = max(demand_now, demand_next)
        desired_ratio = float(
            np.clip(effective_demand / max(self.target_util, 1e-8), 0.125, 1.0)
        )

        # Nếu thiếu tài nguyên hoặc SLA xấu thì wake/boot.
        if sla_ratio > 0.015 or desired_ratio > active_ratio + 0.10:
            if sleep_ratio > 0.0:
                return 1, None  # WAKE_ONE
            if off_ratio > 0.0:
                return 6, None  # BOOT_ONE
            if dvfs_ratio < 0.98 and sla_ratio > 0.005:
                return 3, None  # DVFS_UP
            return 0, None

        # Nếu dư tài nguyên rõ ràng thì sleep bớt host.
        if desired_ratio < active_ratio - 0.18:
            if active_ratio > 0.35:
                return 2, None  # SLEEP_ONE

        # Chỉ giảm DVFS khi tải thấp rõ ràng.
        if demand_now < 0.35 and dvfs_ratio > 0.60:
            return 4, None  # DVFS_DOWN

        return 0, None


def run_policy(env, policy) -> EpisodeMetrics:
    obs, _ = env.reset(seed=42)
    done = False
    total_reward = 0.0
    total_energy = 0.0
    total_it_energy = 0.0
    powers = []
    pues = []
    slas = []
    active_hosts = []
    sleep_hosts = []
    off_hosts = []
    temps = []
    switches = 0
    migrations = 0
    migration_cost = 0.0

    while not done:
        if hasattr(policy, "policy"):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action, _ = policy.predict(obs)

        action = int(np.asarray(action).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_energy += float(info["power_total"])
        total_it_energy += float(info["power_it"])
        powers.append(float(info["power_total"]))
        pues.append(float(info["pue"]))
        slas.append(float(info["sla_violation"]))
        active_hosts.append(int(info["active_hosts"]))
        sleep_hosts.append(int(info["sleep_hosts"]))
        off_hosts.append(int(info["off_hosts"]))
        temps.append(float(info["avg_temp"]))
        switches += int(info["switches"])
        migrations += int(info["migrations"])
        migration_cost += float(info.get("migration_cost", 0.0))

    return EpisodeMetrics(
        total_reward=float(total_reward),
        total_energy=float(total_energy),
        total_it_energy=float(total_it_energy),
        avg_power=float(np.mean(powers)) if powers else 0.0,
        avg_pue=float(np.mean(pues)) if pues else 0.0,
        sla_rate=float(np.mean(slas)) if slas else 0.0,
        avg_active_hosts=float(np.mean(active_hosts)) if active_hosts else 0.0,
        avg_sleep_hosts=float(np.mean(sleep_hosts)) if sleep_hosts else 0.0,
        avg_off_hosts=float(np.mean(off_hosts)) if off_hosts else 0.0,
        avg_temp=float(np.mean(temps)) if temps else 0.0,
        total_switches=int(switches),
        total_migrations=int(migrations),
        total_migration_cost=float(migration_cost),
    )