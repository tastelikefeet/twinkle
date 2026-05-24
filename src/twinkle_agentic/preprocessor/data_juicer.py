# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.preprocessor import Preprocessor
from typing import Any, Dict, List


class DataJuicerPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        from data_juicer.core.data import NestedDataset
        from data_juicer.ops.filter import TextLengthFilter
        from data_juicer.ops.mapper import WhitespaceNormalizationMapper
