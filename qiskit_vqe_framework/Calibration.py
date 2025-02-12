from __future__ import annotations
import abc
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
import copy
import os
import pickle

class Calibration(metaclass=abc.ABCMeta):
    """Abstract base class for all calibration classes
    """
    def __init__(self,
                 name: str,
                 **kwargs) -> None:
        """
        Args:
            name: Name of the calibration class, e.g., AnsatzCalibration for calibration class that determines the VQE Ansatz.
        """
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
        """
        Returns:
            Calibration class properties as a dictionary.
        """
        return copy.deepcopy(self.__dict__)

    def to_pickle(self,
                  fname: str):
        """Saves Calibration class in a serialized pickle file.

        Args:
            fname: Name of the file to which the object should be saved

        Raises:
            ValueError: if the file already exists.
        """
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))

        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @abc.abstractmethod
    def get_filevector(self) -> Tuple[List, List]:
        """Define method to write the summarized (shorted) calibration data in a list format

        Returns:
            Header strings and the corresponding list of data as a tuple.
        """


