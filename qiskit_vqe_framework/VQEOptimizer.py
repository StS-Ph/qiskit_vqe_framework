from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from . import TerminationChecker as tc
from . import Calibration as cal
import qiskit.algorithms.optimizers as optimizers
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.primitives import BaseEstimator
# from qiskit.algorithms.gradients import DerivativeType
import copy
import os
import pickle
import yaml

class OptimizerCalibration(cal.Calibration):
    def __init__(self,
                 name_str: str,
                 maxiter: int,
                 grad_meth: str,
                 param_map_init: Union[Sequence[float], Dict[str, float], None] = None,
                 termination_checker: Union[tc.TerminationChecker, None] = None) -> None:
        super().__init__("OptimizerCalibration")
        self.optimizer_name = name_str

        if maxiter > 0:
            self.maxiter = maxiter
        else:
            raise ValueError("maxiter must be a positive non-zero integer!")

        self.grad_meth = grad_meth

        self._param_map_init = param_map_init

        if self.param_map_init is None:
            self._use_custom_param_init = False
        else:
            self._use_custom_param_init = True

        self.termination_checker = termination_checker
    
    @property
    def param_map_init(self):
        return self._param_map_init
    @param_map_init.setter
    def param_map_init(self,
                       param_map: Union[Sequence[float], Dict[str, float], None]):
        self._param_map_init = param_map

        if param_map is None:
            self._use_custom_param_init = False
        else:
            self._use_custom_param_init = True

    @property
    def use_custom_param_init(self):
        return self._use_custom_param_init

    def __repr__(self):
        out = "OptimizerCalibration(name_str={}, maxiter={}, grad_meth={}, param_map_init={}, termination_checker={})".format(self.optimizer_name, self.maxiter, self.grad_meth, self.param_map_init, self.termination_checker)

        return out

    def to_dict(self):
        opt_cal_dict = super().to_dict()
        param_map_init = opt_cal_dict.pop("_param_map_init")
        opt_cal_dict["param_map_init"] = param_map_init

        use_custom_param_init = opt_cal_dict.pop("_use_custom_param_init")
        opt_cal_dict["use_custom_param_init"] = use_custom_param_init

        return opt_cal_dict

    def to_yaml(self,
                fname: str):
        opt_cal_dict = self.to_dict()
        term_checker = opt_cal_dict.pop("termination_checker", None)
        if term_checker is not None:
            opt_cal_dict["termination_checker"] = term_checker.to_dict()

        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))
        
        with open(fname, "w") as f:
            yaml.dump(opt_cal_dict, f)

    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

        header = []
        data = []

        header.append("optimizer")
        data.append(self.optimizer_name)

        header.append("opt_max_iter")
        data.append(self.maxiter)

        header.append("grad_method")
        data.append(self.grad_meth)

        header.append("use_custom_param_init")
        data.append(self.use_custom_param_init)

        header.append("termination_checker")
        if self.termination_checker is None:
            data.append("None")
        else:
            data.append(self.termination_checker.name)

        return header, data
    
def get_OptimizerCalibration_from_dict(opt_cal_dict: dict) -> OptimizerCalibration:
    
    name_str = opt_cal_dict.pop("optimizer_name", None)
    if name_str is None:
        raise ValueError("could not retrieve optimizer name from file!")
    maxiter = opt_cal_dict.pop("maxiter", None)
    if maxiter is None:
        raise ValueError("could not retrieve maximal number of iterations from file!")
    grad_meth = opt_cal_dict.pop("grad_meth", None)
    if grad_meth is None:
        raise ValueError("could not retrieve gradient method from file!")

    name = opt_cal_dict.pop("name", None)
    use_custom_param_init = opt_cal_dict.pop("use_custom_param_init", None)
    #print(ansatz_cal_dict)

    opt_cal = OptimizerCalibration(name_str, maxiter, grad_meth, **opt_cal_dict)
    return opt_cal

def get_OptimizerCalibration_from_yaml(fname: str) -> OptimizerCalibration:
    
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    opt_cal_dict = None
    raw_data = None
    with open(fname, "r") as f:
        raw_data = f.read()

    opt_cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
    if opt_cal_dict is None:
        raise ValueError("Something went wrong while reading in yml text file! resulting dictionary is empty!")
    
    term_checker_dict = opt_cal_dict.pop("termination_checker")
    if term_checker_dict is not None:
        term_checker_name = term_checker_dict.pop("name")
        term_checker = tc.get_termination_checker_from_name(term_checker_name, term_checker_dict)
        opt_cal_dict["termination_checker"] = term_checker

    return get_OptimizerCalibration_from_dict(opt_cal_dict)

def get_OptimizerCalibration_from_pickle(fname: str) -> OptimizerCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    opt_cal = None
    with open(fname, "rb") as f:
        opt_cal = pickle.load(f)

    if not isinstance(opt_cal, OptimizerCalibration):
        raise ValueError("loaded pickle object is no OptimizerCalibration!")

    return opt_cal


class VQEOptimizer:
    def __init__(self,
                 optimizer_parameters: OptimizerCalibration) -> None:
        self._parameters = optimizer_parameters
        self._optimizer = self._get_optimizer()
        #self._parameters_updated = False

    @property
    def parameters(self):
        return self._parameters
    @parameters.setter
    def parameters(self,
                   new_parameters: OptimizerCalibration):
        self._parameters = new_parameters
        self._update_optimizer()

    @property
    def optimizer(self):
        return self._optimizer

    def __repr__(self):
        out = "VQEOptimizer(optimizer_parameters={})".format(self.parameters)
        return out

    def to_dict(self):
        optimizer_dict = {}
        optimizer_dict["parameters"] = self.parameters.to_dict()
        optimizer_dict["optimizer"] = self.optimizer

        return optimizer_dict

    def update_parameters(self,
                          new_parameters: OptimizerCalibration) -> None:
        self.parameters = new_parameters

    def _update_optimizer(self) -> None:
        self._optimizer = self._get_optimizer()

    def _get_optimizer(self) -> optimizers.optimizer.Optimizer:
        if self.parameters.optimizer_name == "SPSA":
            if self.parameters.grad_meth != "fin_diff":
                raise ValueError("assigned gradient method string {} is not compatible with {} optimizer, since finite difference gradient is intrinsically used!".format(self.parameters.grad_meth, self.parameters.optimizer_name))
            return optimizers.SPSA(maxiter = self.parameters.maxiter, termination_checker=self.parameters.termination_checker)
        else:
            raise ValueError("optimizer name string {} does not match any supported optimizer class!".format(self.parameters.optimizer_name))
                 

    def get_gradient(self,
                     estimator: BaseEstimator,
                     options: Union[Dict, None] = None,
                     derivative_type: None = None) -> Union[BaseEstimatorGradient, None]:
        # To-do: implement gradient objects properly
        return None
        
