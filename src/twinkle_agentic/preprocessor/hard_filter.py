# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.preprocessor import Preprocessor
from typing import Any, Dict, List

from twinkle.data_format import Trajectory

class HardFilter(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.hard_filter(rows)
        rows = self.map_row_to_col(rows)
        return rows
    
    def hard_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...
