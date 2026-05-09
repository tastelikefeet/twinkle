from abc import abstractmethod

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunks


class ModelCondenser(Condenser):

    @abstractmethod
    def __call__(self, chunks: Chunks, **kwargs) -> Chunks:
        pass