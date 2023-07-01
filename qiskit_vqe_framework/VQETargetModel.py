from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.opflow import PauliSumOp
from . import Calibration as cal
import copy
import abc

class ModelCalibration(cal.Calibration):
    def __init__(self,
                 model_name: str,
                 **kwargs) -> None:
        super().__init__("ModelCalibration", model_name=model_name, **kwargs)

    def __repr__(self):
        string_list = []
        attr_dict = self.to_dict()
        cal_name = attr_dict.pop("name", None)
        for k, v in attr_dict.items():
            string_list.append(f"{k}={v}")
            
        out = "ModelCalibration(%s)" % ", ".join(string_list)

        return out

    def to_dict(self) -> Dict:
        attr_dict = super().to_dict()
        #cal_name = attr_dict.pop("name", None)

        return attr_dict
    
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

        header = []
        data = []

        attr_dict = self.to_dict()
        cal_name = attr_dict.pop("name", None)

        for key, val in attr_dict.items():
            header.append(key)
            data.append(val)

        return header, data

        
class VQETargetModel:
    def __init__(self,
                 model_parameters: ModelCalibration) -> None:
        self._validate_parameters(model_parameters)
        self._parameters = model_parameters
        self._hamiltonian = self._get_hamiltonian()
        self._aux_ops = self._get_aux_ops()

    @property
    def parameters(self):
        return self._parameters
    @parameters.setter
    def parameters(self,
                   new_model_parameters: ModelCalibration):
        self._validate_parameters(new_model_parameters)
        self._parameters = new_model_parameters
        self._update_operators()

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def aux_ops(self):
        return self._aux_ops

    def __repr__(self):
        out = "VQETargetModel(model_parameters={})".format(self.parameters)
        return out
    
    def to_dict(self):
        model_dict = {}
        model_dict["parameters"] = self.parameters.to_dict()
        model_dict["hamiltonian"] = self.hamiltonian
        model_dict["aux_ops"] = self.aux_ops

        return model_dict

    def update_parameters(self,
                          new_model_parameters: MFSCalibration) -> None:
        self.parameters = new_model_parameters

    def _update_operators(self) -> None:
        self._hamiltonian = self._get_hamiltonian()
        self._aux_ops = self._get_aux_ops()

    @abc.abstractmethod
    def _validate_parameters(self,
                             model_parameters: ModelCalibration) -> None:
        """
        Define method to validate model parameters (raise ValueErrors if not valid)
        """

    @abc.abstractmethod
    def _get_hamiltonian(self) -> Union[PauliSumOp, SparsePauliOp]:
        """
        Define method to generate hamiltonian from model parameters
        """

    @abc.abstractmethod
    def _get_aux_ops(self) -> Union[Dict[str, Union[PauliSumOp, SparsePauliOp]], None]:
        """
        Define method to generate all observables that should be additionally measured (to Hamiltonian) in Dict format {"name": observable}. Return None if no additional observables should be measured.
        """

    @abc.abstractmethod
    def get_ed_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]:
        """
        Define method to generate energy penalty term in exact diagonalization. If no penalty, then return None
        """

    @abc.abstractmethod
    def get_vqe_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]:
        """
        Define method to generate energy penalty term in vqs. If no penalty, then return None
        """
        
class TransverseFieldIsingModel(VQETargetModel):
    def __init__(self,
                 num_spins: int,
                 J: float = 1.0,
                 g: float = -0.5) -> None:
        tfim_cal = ModelCalibration("transverse_field_Ising_model", num_spins=num_spins, J=J, g=g)
        super().__init__(tfim_cal)

    def _validate_parameters(self,
                             cal: ModelCalibration) -> None:
        if not isinstance(cal.num_spins, int):
            raise ValueError("number of spins must be integer!")

    def _get_hamiltonian(self) -> Union[PauliSumOp, SparsePauliOp]:
        J = self.parameters.J
        g = self.parameters.g
        L = self.parameters.num_spins

        H = PauliSumOp(SparsePauliOp("ZZ"+"I"*(L-2), J))
        for l in range(1,L-1):
            IL = "I" * l
            IR = "I" * (L-2-l)
            H = H.add(PauliSumOp(SparsePauliOp(IL+"ZZ"+IR, J)))

        H = H.add(PauliSumOp(SparsePauliOp("X"+("I" * (L-1)), g)))
        for l in range(1,L):
            # generate Pauli string
            IL = "I" * l
            IR = "I" * (L-1-l)
            H =H.add(PauliSumOp(SparsePauliOp(IL+"X"+IR,g)))

        return H

    def _get_aux_ops(self) -> Union[Dict[str, Union[PauliSumOp, SparsePauliOp]], None]:
        L = self.parameters.num_spins
        qtot = PauliSumOp(SparsePauliOp("Z"+("I" * (L-1)), 1/2))
        for l in range(1,L):
            # generate Pauli string
            IL = "I" * l
            IR = "I" * (L-1-l)
            qtot =qtot.add(PauliSumOp(SparsePauliOp(IL+"Z"+IR,1/2)))
        
        aux_ops = {'qtot': qtot}

        return aux_ops

    def get_ed_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]:
        return None

    def get_vqe_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]:
        return None

    
