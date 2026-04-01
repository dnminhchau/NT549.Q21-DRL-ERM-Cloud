from __future__ import annotations

from dataclasses import dataclass
from email import policy

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
    """Luân phiên mở thêm active host khi tải cao, giảm dần khi tải thấp."""

    def __init__(self):
        self.cursor = 0

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        active_ratio = float(obs[2])
        sleep_ratio = float(obs[8])
        off_ratio = float(obs[9])
        temp_ratio = float(obs[11])
        self.cursor = (self.cursor + 1) % 4

        if demand_now > 0.90:
            return (1 if self.cursor % 2 == 0 else 6), None
        if demand_now > 0.75 and active_ratio < 0.95:
            return 1, None
        if temp_ratio > 0.78:
            return 4, None
        if demand_now < 0.25 and sleep_ratio > 0.15:
            return 5, None
        if demand_now < 0.40 and active_ratio > 0.25:
            return 2, None
        if demand_now < 0.55 and off_ratio < 0.40:
            return 4, None
        return 0, None

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
            if sleep_ratio > 0.10:
                return 5, None
        return 0, None



class BestFitLikePolicy:
    def __init__(self, target_util: float = 0.75):
        self.target_util = target_util

    def predict(self, obs: np.ndarray):
        demand_now = float(obs[0])
        active_ratio = float(obs[2])
        dvfs_ratio = float(obs[3])
        pue_ratio = float(obs[10]) * 3.0
        temp_ratio = float(obs[11])
        migration_ratio = float(obs[13])

        desired_ratio = min(max(demand_now / max(self.target_util, 1e-8), 0.125), 1.0)

        if desired_ratio > active_ratio + 0.08:
            return 1, None
        if desired_ratio < active_ratio - 0.08 and active_ratio > 0.2:
            return 2, None
        if demand_now > 0.85 and dvfs_ratio < 0.99:
            return 3, None
        if demand_now < 0.40 and dvfs_ratio > 0.55:
            return 4, None
        if demand_now < 0.25 and pue_ratio > 1.28:
            return 5, None
        if temp_ratio > 0.82 or migration_ratio > 0.45:
            return 4, None
        return 0, None


def run_policy(env, policy) -> EpisodeMetrics:
    obs, _ = env.reset()
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
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
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

