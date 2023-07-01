import abc
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Union

class TerminationChecker(metaclass=abc.ABCMeta):
    def __init__(self,
                 buffer_length: int,
                 name_str: str) -> None:
        if buffer_length <= 0:
            raise ValueError("length of history buffer {} must be a positive integer!".format(buffer_length))
        self.buffer_length = buffer_length
        self.values = []
        self.name = name_str

    def __call__(self,
                 nfev: int,
                 parameters: Sequence[float],
                 value: float,
                 stepsize: int,
                 accepted: bool) -> bool:
        # add current value to history buffer
        self.values.append(value)
        # check current history buffer length
        if len(self.values) > self.buffer_length:
            # values should be of length buffer_length, containing the latest values
            self.values = self.values[-self.buffer_length:]

        # call termination checker if values list is large enough
        if len(self.values) == self.buffer_length:
            return self._check_termination(nfev, parameters, value, stepsize, accepted)
        # otherwise return false
        return False
            
    @abc.abstractmethod
    def _check_termination(self,
                          nfev: int,
                          parameters: Sequence[float],
                          value: float,
                          stepsize: int,
                          accepted: bool) -> bool:
        """
        Define method to determine convergence of value
        """

class RelativeEnergyChecker(TerminationChecker):
    def __init__(self,
                 buffer_length: int,
                 considered_values_length: int,
                 epsilon: float) -> None:
        if epsilon < 0.0:
            raise ValueError("tolerance for termination check {} must be non-negative!".format(epsilon))
        self.epsilon = epsilon
        
        if considered_values_length <= 0:
            raise ValueError("minimal number of values that have to be considered for relative change {} must be non-negative!".format(considered_values_length))
        if considered_values_length > buffer_length:
            raise ValueError("minimal number of values that have to be considered for relative change {} must be smaller or equal to the history buffer length!".format(considered_values_length))
        self.considered_values_length = considered_values_length
        
        super().__init__(buffer_length, "relative_energy_change")

    def _check_termination(self,
                          nfev: int,
                          parameters: Sequence[float],
                          value: float,
                          stepsize: int,
                          accepted: bool) -> bool:
        # handle optimization steps that have not been accepted by the optimizer routine
        if not accepted:
            return False
        # calculate relative change in the values history
        dv = self._calc_relative_change()
        # check if the mean relative change is below epsilon
        if len(dv) >= self.considered_values_length and np.average(dv) <= self.epsilon:
            return True

        #otherwise return False
        return False

    def _calc_relative_change(self) -> Sequence[float]:
        relative_change = []
        # filter all values that are too close to zero
        considered_values = list(filter(lambda e: np.abs(e) > 1e-05, self.values))
        considered_values = considered_values
        # calculate relative change
        for old, new in zip(considered_values[:], considered_values[1:]):
            delta = np.abs((new-old)/new)
            relative_change.append(delta)

        return relative_change[-self.considered_values_length:]

    def __repr__(self):
        out = "RelativeEnergyChecker(buffer_length={}, considered_values_length={}, epsilon={})".format(self.buffer_length, self.considered_values_length, self.epsilon)
        return out

class LinearFitChecker(TerminationChecker):
    def __init__(self,
                 buffer_length: int,
                 epsilon: float) -> None:
        if epsilon < 0.0:
            raise ValueError("tolerance for termination check {} must be non-negative!".format(epsilon))
        self.epsilon = epsilon
        
        super().__init__(buffer_length, "linear_fit")

    def _check_termination(self,
                          nfev: int,
                          parameters: Sequence[float],
                          value: float,
                          stepsize: int,
                          accepted: bool) -> bool:
        # handle optimization steps that have not been accepted by the optimizer routine
        if not accepted:
            return False
        # linear fit to current values history buffer
        pp = np.polyfit(range(self.buffer_length), self.values, 1)
        slope = pp[0]/self.buffer_length
        # check if the mean relative change is below epsilon
        if np.abs(slope) <= self.epsilon:
            return True

        #otherwise return False
        return False

    def __repr__(self):
        out = "LinearFitChecker(buffer_length={}, epsilon={})".format(self.buffer_length, self.epsilon)
        return out
