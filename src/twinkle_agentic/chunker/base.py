from abc import ABC, abstractmethod

from twinkle.data_format import Trajectory


class Chunker(ABC):

    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> Chunks:
        raise NotImplementedError