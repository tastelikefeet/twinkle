# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List

from twinkle.data_format import Trajectory


class Preprocessor:

    def __call__(self, rows: List[Dict]) -> List[Trajectory]:
        ...


class DataFilter:

    def __call__(self, row) -> bool:
        ...
