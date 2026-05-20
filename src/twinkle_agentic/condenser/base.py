from abc import ABC, abstractmethod

from twinkle_agentic.data_format import Chunks


class Condenser(ABC):

    @abstractmethod
    def __call__(self, chunks: Chunks, **kwargs) -> Chunks:
        raise NotImplementedError
