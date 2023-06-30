from __future__ import annotations
import abc
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
import copy

class Calibration(metaclass=abc.ABCMeta):
    def __init__(self,
                 name: str,
                 **kwargs) -> None:
        self.name = name

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self) -> str:
        string_list = []
        for k, v in self.__dict__.items():
            string_list.append(f"{k}={v}")
            
        out = "Calibration(%s)" % ", ".join(string_list)
        return out

    def to_dict(self) -> Dict:
        return copy.copy(self.__dict__)

    @abc.abstractmethod
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """


