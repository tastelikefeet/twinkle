from abc import ABC, abstractmethod

from twinkle.data_format import Trajectory


class Rollout(ABC):

    @abstractmethod
    def __call__(self, trajectory: Trajectory, **kwargs) -> Trajectory:
        raise NotImplementedError()
