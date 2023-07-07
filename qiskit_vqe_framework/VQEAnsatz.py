from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Instruction
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from . import Calibration as cal
import abc
import yaml
import pickle
import os

class AnsatzCalibration(cal.Calibration):
    def __init__(self,
                 num_qubits: int,
                 num_layers: int,
                 ansatz_str: str,
                 psi_start: Union[str, list, None] = None,
                 **kwargs) -> None:
        super().__init__("AnsatzCalibration")
        
        if num_qubits > 0:
            self.num_qubits = num_qubits
        else:
            raise ValueError("number of qubits must be a postive non-zero integer!")

        if num_layers > 0:
            self.num_layers = num_layers
        else:
            raise ValueError("number of layers must be a postive non-zero integer!")

        if ansatz_str:
            self.ansatz_str = ansatz_str
        else:
            raise ValueError("ansatz_str must be non-empty string!")

        self._psi_start = psi_start

        if self.psi_start is None:
            self._use_custom_state_init = False
        else:
            if len(self.psi_start) == 0:
                self._use_custom_state_init = False
            else:
                self._use_custom_state_init = True

        for key, val in kwargs.items():
            setattr(self, key, val)
            

    @property
    def psi_start(self):
        return self._psi_start
    @psi_start.setter
    def psi_start(self,
                  vector: Union[list, None]):
        self._psi_start = vector

        if vector is None:
            self._use_custom_state_init = False
        else:
            self._use_custom_state_init = True

    @property
    def use_custom_state_init(self):
        return self._use_custom_state_init

    def __repr__(self):
        ansatz_cal_dict = self.to_dict()
        cal_name = ansatz_cal_dict.pop("name")
        use_custom_state_init = ansatz_cal_dict.pop("use_custom_state_init")

        string_list = []
        for k,v in ansatz_cal_dict.items():
            string_list.append("{}={}".format(k,v))
        out = "AnsatzCalibration(%s)" % ", ".join(string_list)

        return out

    def to_dict(self):
        qalg_cal_dict = super().to_dict()
        psi_start = qalg_cal_dict.pop("_psi_start")
        qalg_cal_dict["psi_start"] = psi_start

        use_custom_state_init = qalg_cal_dict.pop("_use_custom_state_init")
        qalg_cal_dict["use_custom_state_init"] = use_custom_state_init

        return qalg_cal_dict
    
    def to_yaml(self,
                fname: str):
        ansatz_cal_dict = self.to_dict()
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))
        
        with open(fname, "w") as f:
            yaml.dump(ansatz_cal_dict, f)
    
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

        header = []
        data = []

        header.append("num_qubits")
        data.append(self.num_qubits)

        header.append("num_layers")
        data.append(self.num_layers)

        header.append("ansatz")
        data.append(self.ansatz_str)

        header.append("use_custom_state_init")
        data.append(self.use_custom_state_init)
                
        return header, data
    
def get_AnsatzCalibration_from_yaml(fname: str) -> AnsatzCalibration:
    
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    ansatz_cal_dict = None
    raw_data = None
    with open(fname, "r") as f:
        raw_data = f.read()

    ansatz_cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
    if ansatz_cal_dict is None:
        raise ValueError("Something went wrong while reading in yml text file! resulting dictionary is empty!")

    num_qubits = ansatz_cal_dict.pop("num_qubits", None)
    if num_qubits is None:
        raise ValueError("could not retrieve number of qubits from file!")
    num_layers = ansatz_cal_dict.pop("num_layers", None)
    if num_layers is None:
        raise ValueError("could not retrieve number of layers from file!")
    ansatz_str = ansatz_cal_dict.pop("ansatz_str", None)
    if ansatz_str is None:
        raise ValueError("could not retrieve ansatz_str from file!")

    name = ansatz_cal_dict.pop("name", None)
    use_custom_state_init = ansatz_cal_dict.pop("use_custom_state_init", None)
    #print(ansatz_cal_dict)

    ansatz_cal = AnsatzCalibration(num_qubits, num_layers, ansatz_str, **ansatz_cal_dict)
    return ansatz_cal

def get_AnsatzCalibration_from_pickle(fname: str) -> AnsatzCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    ansatz_cal = None
    with open(fname, "rb") as f:
        ansatz_cal = pickle.load(f)

    if not isinstance(ansatz_cal, AnsatzCalibration):
        raise ValueError("loaded pickle object is no AnsatzCalibration!")

    return ansatz_cal

class VQEAnsatz:
    def __init__(self,
                 ansatz_parameters: AnsatzCalibration) -> None:
        self._parameters = ansatz_parameters
        self._circuit = self._get_circuit()

    @property
    def parameters(self):
        return self._parameters
    @parameters.setter
    def parameters(self,
                   new_ansatz_parameters: AnsatzCalibration):
        self._parameters = new_ansatz_parameters
        self._update_circuit()

    @property
    def circuit(self):
        return self._circuit

    def update_parameters(self,
                          new_ansatz_parameters: QAlgCalibration) -> None:
        self.parameters = new_ansatz_parameters

    def _update_circuit(self) -> None:
        self._circuit = self._get_circuit()

    @abc.abstractmethod
    def _get_circuit(self) -> QuantumCircuit:
        """
        Define method to generate the vqe ansatz circuit
        """

    def __repr__(self):
        out = "VQEAnsatz(parameters={})".format(self.parameters)

        return out
    def to_dict(self):
        ansatz_dict = {}
        ansatz_dict["parameters"] = self._parameters.to_dict()
        ansatz_dict["circuit"] = self._circuit

        return ansatz_dict

class ESU2(VQEAnsatz):
    def __init__(self,
                 num_qubits: int,
                 reps: int = 3,
                 su2_gates: Union[str, Instruction, QuantumCircuit, List[Union[str, Instruction, QuantumCircuit]], None] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "reverse_linear",
                 initial_state: Union[str, list, None]= None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = "p",
                 insert_barriers: bool = False) -> None:
        if initial_state is not None:
            intial_state_valid, type_valid, size_valid = self._validate_initial_state(num_qubits, initial_state)
            if not intial_state_valid:
                if not type_valid:
                    raise TypeError("Data type of intial state is not valid!")
                if not size_valid:
                    raise ValueError("size of initial state is not valid!")
                else:
                    raise ValueError("initial state is not valid!")
            
        esu2_cal = AnsatzCalibration(num_qubits, reps, "ESU2", psi_start=initial_state, su2_gates = su2_gates, entanglement = entanglement, skip_unentangled_qubits = skip_unentangled_qubits, skip_final_rotation_layer = skip_final_rotation_layer, parameter_prefix = parameter_prefix, insert_barriers = insert_barriers)
        super().__init__(esu2_cal)

     #This function validates if a given initial state for the quantum circuit is of correct type and size
    def _validate_initial_state(self,
                                num_qubits: int,
                                psi_init: Union[str, list]) -> Tuple[bool, bool, bool]:
        type_is_valid = False
        size_is_valid = False
        state_is_valid = False
        if isinstance(psi_init, str):
            type_is_valid = True
            state_length = len(psi_init)
            valid_length = num_qubits
        elif isinstance(psi_init, list):
            type_is_valid = True
            state_length = len(psi_init)
            valid_length = 2**num_qubits
    
        if type_is_valid and state_length == valid_length:
            size_is_valid = True
            state_is_valid = True
        
    
        return state_is_valid, type_is_valid, size_is_valid

    def _get_circuit(self) -> QuantumCircuit:
        circ_init_state = QuantumCircuit(self.parameters.num_qubits)

        if self.parameters.psi_start is not None:
            circ_init_state.initialize(self.parameters.psi_start)
        else:
            circ_init_state = None
            
        circ_su2 = QuantumCircuit(self.parameters.num_qubits)

        esu2_ansatz = EfficientSU2(num_qubits=self.parameters.num_qubits, reps=self.parameters.num_layers, su2_gates=self.parameters.su2_gates, entanglement=self.parameters.entanglement, initial_state=circ_init_state, skip_unentangled_qubits = self.parameters.skip_unentangled_qubits, skip_final_rotation_layer=self.parameters.skip_final_rotation_layer, parameter_prefix = self.parameters.parameter_prefix, insert_barriers=self.parameters.insert_barriers).decompose()

        circ_su2.compose(esu2_ansatz, inplace=True)

        #circ_su2.decompose()

        return circ_su2
