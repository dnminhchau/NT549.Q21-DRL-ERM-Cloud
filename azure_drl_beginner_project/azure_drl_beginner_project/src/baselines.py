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


class FixedPolicy:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs: np.ndarray):
        return self.action, None


class RoundRobinPolicy:
    """
    Baseline heuristic kiểu Round Robin ở mức cluster:
    - tải tăng thì ưu tiên wake/boot luân phiên
    - tải giảm thì ưu tiên sleep/power off luân phiên
    """

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
        temp_ratio = float(obs[11])

        self.cursor = (self.cursor + 1) % 4

        rising = demand_next > demand_now + 0.03
        high_load = demand_now > 0.82 or rising
        very_high_load = demand_now > 0.92 or sla_ratio > 0.03

        if very_high_load:
            if off_ratio > 0.0 and self.cursor % 2 == 0:
                return 6, None
            return 1, None

        if high_load and active_ratio < 0.95:
            return 1, None

        if demand_now > 0.78 and dvfs_ratio < 0.98 and temp_ratio < 0.80:
            return 3, None

        if temp_ratio > 0.82:
            return 4, None

        if demand_now < 0.30 and active_ratio > 0.25:
            if sleep_ratio > 0.20 and self.cursor % 2 == 1:
                return 5, None
            return 2, None

        if demand_now < 0.45 and dvfs_ratio > 0.60:
            return 4, None

        return 0, None


class ThresholdPolicy:
    """
    Baseline ngưỡng đơn giản:
    - phản ứng chủ yếu theo demand/SLA
    - ít quan tâm nhiệt độ và migration hơn BestFit
    """

    def __init__(self, high: float = 0.80, low: float = 0.32):
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
                    return 1, None
                if off_ratio > 0.0:
                    return 6, None
                return 1, None
            if dvfs_ratio < 0.98 and sla_ratio > 0.01:
                return 3, None

        if demand_now < self.low:
            if dvfs_ratio > 0.60:
                return 4, None
            if active_ratio > 0.30:
                return 2, None
            if sleep_ratio > 0.15:
                return 5, None

        return 0, None


class BestFitPolicy:
    """
    Heuristic gần với Best Fit ở mức cluster:
    - ước lượng số active host mong muốn từ demand / target_util
    - ưu tiên gom tải vừa đủ để giảm overprovision
    - có xét thêm PUE, nhiệt độ và migration
    """

    def __init__(self, target_util: float = 0.78):
        self.target_util = target_util

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        demand_next = float(obs[1])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        power_ratio = float(obs[4])
        sla_ratio = float(obs[5])
        sleep_ratio = float(obs[8])
        off_ratio = float(obs[9])
        pue_ratio = float(obs[10]) * 3.0
        temp_ratio = float(obs[11])
        migration_ratio = float(obs[13])

        effective_demand = max(demand_now, 0.65 * demand_now + 0.35 * demand_next)
        desired_ratio = np.clip(effective_demand / max(self.target_util, 1e-8), 0.125, 1.0)

        if desired_ratio > active_ratio + 0.10 or sla_ratio > 0.025:
            if sleep_ratio > 0.0:
                return 1, None
            if off_ratio > 0.0:
                return 6, None
            if dvfs_ratio < 0.98 and temp_ratio < 0.82:
                return 3, None
            return 0, None

        if desired_ratio < active_ratio - 0.12:
            if dvfs_ratio > 0.60 and demand_now < 0.55:
                return 4, None
            if active_ratio > 0.25:
                return 2, None
            if sleep_ratio > 0.15 and pue_ratio > 1.22:
                return 5, None

        if temp_ratio > 0.84:
            if active_ratio < 0.95:
                return 1, None
            return 4, None

        if migration_ratio > 0.45 and dvfs_ratio > 0.60:
            return 4, None

        if demand_now > 0.94 and dvfs_ratio < 0.98 and temp_ratio < 0.80 and sla_ratio > 0.005:
            return 3, None

        if demand_now < 0.38 and power_ratio > 0.22 and dvfs_ratio > 0.60:
            return 4, None

        return 0, None


BestFitLikePolicy = BestFitPolicy


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
    )