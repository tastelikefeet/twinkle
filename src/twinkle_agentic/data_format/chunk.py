import sys
from dataclasses import dataclass
from typing import Union, Literal, Any, List

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Chunk(TypedDict, total=False):

    type: Literal['text', 'image', 'video', 'audio']
    content: Union[str, Any]
    raw: Union[str, Any]
    role: str


@dataclass
class Chunks:

    chunks: List[Chunk]

    def to_trajectory(self):
        ...
