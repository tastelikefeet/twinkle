from twinkle.data_format import Trajectory
from .base import Rollout

class MultiTurnRollout(Rollout):

    def __call__(self, trajectory: Trajectory, **kwargs) -> Trajectory:
        