"""Per-job context passed from the orchestrator to every stage."""
import time
from dataclasses import dataclass, field
from pathlib import Path

from train import common


@dataclass
class JobContext:
    job: str
    job_dir: Path
    profile: dict                     # resolved profile (sizes, budgets, eval episodes)
    budget: dict                      # this job's budget (steps / epochs / envs)
    log_fn: object = print
    started: float = field(default_factory=time.time)
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.job_dir = Path(self.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = common.MetricLog(self.job)
        self.ckpt = common.Checkpointer(self.job_dir, every_s=self.profile.get('ckpt_every_s', 600))
        self.throttle = common.ThroughputMonitor(enabled=self.profile.get('throttle_guard', False))

    def log(self, msg):
        self.log_fn(f'[{self.job}] {msg}')

    def note(self, msg):
        self.notes.append(msg)
        self.log(msg)
