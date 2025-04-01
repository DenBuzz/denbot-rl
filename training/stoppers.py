from typing import Dict

from ray.tune.stopper.stopper import Stopper


class CurriculumStopper(Stopper):
    def __call__(self, trial_id: str, result: Dict):
        return bool(result.get("curriculum_complete", False))

    def stop_all(self):
        return False
