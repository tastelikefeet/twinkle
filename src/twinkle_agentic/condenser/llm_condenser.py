from twinkle.data_format import Trajectory
from twinkle.sampler import vLLMSampler
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunks


class LLMCondenser(Condenser):

    def __init__(self, model_id, **kwargs):
        self.sampler = vLLMSampler(model_id, **kwargs)

    def condense(self, chunks: Chunks, system: str = None) -> Chunks:
        ...