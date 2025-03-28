from typing import Dict

from ray.tune.stopper.stopper import Stopper


class TaskStopper(Stopper):
    def __init__(self, max_task: int):
        self._max_task = max_task

    def __call__(self, trial_id: str, result: Dict):
        return result.get("current_task", 0) >= self._max_task

    def stop_all(self):
        return False
