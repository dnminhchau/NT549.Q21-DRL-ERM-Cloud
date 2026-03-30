from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeMetrics:
    total_reward: float
    total_energy: float
    avg_power: float
    avg_pue: float
    avg_temp_c: float
    sla_rate: float
    avg_active_hosts: float
    total_switches: int
    total_migrations: int


class FixedPolicy:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs: np.ndarray):
        return self.action, None


class RoundRobinPolicy:
    """Chu kỳ hành động đơn giản để làm baseline cổ điển."""

    def __init__(self):
        self.idx = 0
        self.actions = [0, 1, 0, 2, 3, 4]

    def predict(self, obs: np.ndarray):
        action = self.actions[self.idx % len(self.actions)]
        self.idx += 1
        return action, None


class ThresholdPolicy:
    def __init__(self, high: float = 0.8, low: float = 0.35):
        self.high = high
        self.low = low

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        sleep_ratio = float(obs[8])

        if demand_now > self.high:
            if active_ratio < 0.99:
                return 1, None
            if dvfs_ratio < 0.99:
                return 3, None
        elif demand_now < self.low:
            if dvfs_ratio > 0.55:
                return 4, None
            if active_ratio > 0.20:
                return 2, None
            if sleep_ratio > 0.05:
                return 5, None
        return 0, None


class BestFitPolicy:
    """Baseline best-fit xấp xỉ theo tỉ lệ host active cần thiết."""

    def __init__(self, target_util: float = 0.75):
        self.target_util = target_util

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        active_ratio = float(obs[2])
        current_dvfs_ratio = float(obs[3])
        sleep_ratio = float(obs[8])

        desired_ratio = min(max(demand_now / max(self.target_util, 1e-8), 0.125), 1.0)

        if desired_ratio > active_ratio + 0.08:
            return 1, None
        if desired_ratio < active_ratio - 0.08 and active_ratio > 0.2:
            return 2, None
        if demand_now > 0.85 and current_dvfs_ratio < 0.99:
            return 3, None
        if demand_now < 0.40 and current_dvfs_ratio > 0.55:
            return 4, None
        if demand_now < 0.30 and sleep_ratio > 0.08:
            return 5, None
        return 0, None


# Giữ lại tên cũ để tương thích notebook cũ
BestFitLikePolicy = BestFitPolicy


def run_policy(env, policy) -> EpisodeMetrics:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    total_energy = 0.0
    powers = []
    pues = []
    temps = []
    slas = []
    active_hosts = []
    switches = 0
    migrations = 0

    while not done:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward
        total_energy += float(info["power"])
        powers.append(float(info["power"]))
        pues.append(float(info.get("pue", 1.0)))
        temps.append(float(info.get("mean_temperature_c", 0.0)))
        slas.append(float(info["sla_violation"]))
        active_hosts.append(int(info["active_hosts"]))
        switches += int(info["switches"])
        migrations += int(info.get("migrations", 0))

    return EpisodeMetrics(
        total_reward=float(total_reward),
        total_energy=float(total_energy),
        avg_power=float(np.mean(powers)) if powers else 0.0,
        avg_pue=float(np.mean(pues)) if pues else 1.0,
        avg_temp_c=float(np.mean(temps)) if temps else 0.0,
        sla_rate=float(np.mean(slas)) if slas else 0.0,
        avg_active_hosts=float(np.mean(active_hosts)) if active_hosts else 0.0,
        total_switches=int(switches),
        total_migrations=int(migrations),
    )
