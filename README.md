# Qiskit VQE data framework

A unpublished project to provide a data framework for running vqe simulations in qiskit.
It provides a calibration class for each major part in a qiskit vqe simualtion (target model, circuit ansatz, optimizer and estimator). Which can be used to easily save all calibration data in seperate files as dictionaries or combined as a filevector in one file.
Each vqe part has additionally a class which contains its calibration class and the corresponding qiskit object. The qiskit objects are generated via methods of this class taking the calibration class as an input. This way calibration and corresponding qiskit object are correctly associated with each other.

Each Calibration class can export its data to a dictionary via the to_dict() function or to a .yaml file via the to_yaml() function. Additionally one can generate a string list with the calibration data via the get_filevector() function. Each calibration class can be either generated from a dictionary via get_*Calibration Class name*_from_dict(d: dict) (inverse of to_dict() function) or from a yaml file via get_*Calibration Class name*_from_yaml(filename: str) or from the pickled Calibration Class object via get_*Calibration Class name*_from_pickle(filename: str).

## Installation

In the current version of this package a lot of changes are made and it is not yet published. Thus it must be installed in editable mode via:
`pip install -e `
in the qiskit_vqe_framework directory.

## Usage

### Ansatz Calibration

To calibrate a VQE ansatz circuit the Calibration class (VQEAnsatz.py) expects 4 input variables: 
- num_qubits: int: number of qubits in the circuit 
- num_layers: int: number of algorithm layers 
- ansatz_str: str: string that names the used ansatz 
- psi_start: Union[str, list, None] = None: the initial state vector 
Other attributes can also be provided based on what the particular ansatz expects via attribute = attr_value.

Note that every VQE ansatz needs to be implemented as a derived class of the VQEAnsatz class. Here additional calibration parameters can be handled appropriately in __init__. The derived class needs to implement a _get_circuit() function which generates the qiskit QuantumCircuit object corresponding to the Ansatz.

### Target Model Calibration

To calibrate the VQE Target Model the calibration class (VQETargetModel.py) only expects 1 input variable, i.e., a certain model name model_name: str. Since all additional parameters depend on the particular model. They can of course also be provided as attribute = attr_value.

Note that every VQE target model needs to be implemented as a derived class of the VQETargetModel class. Here all required calibration parameters can be handled appropriately in __init__. The derived class needs to implement the following abstract methods from VQETargetModel:
- _validate_parameters(model_parameters: ModelCalibration) -> None: Define method to validate model parameters (raise ValueErrors if not valid)
- _get_hamiltonian(self) -> Union[PauliSumOp, SparsePauliOp]: Define method to generate hamiltonian from model parameters
- _get_aux_ops(self) -> Union[Dict[str, Union[PauliSumOp, SparsePauliOp]], None]: Define method to generate all observables that should be additionally measured (to Hamiltonian) in Dict format {"name": observable}. Return None if no additional observables should be measured.
- get_ed_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]: Define method to generate energy penalty term in exact diagonalization. If no penalty, then return None
- get_vqe_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]: Define method to generate energy penalty term in vqs. If no penalty, then return None

### Estimator Calibration

To calibrate the VQE Estimator, the calibration class (VQEEstimator) expects 6 input variables: 
- est_opt: Dict: All Options to calibrate the used estimator class
- noise_model_str: str: Unique name for the used noise model
- coupling_map_str: str: Unique name for the used coupling map
- basis_gates_str: str: Unique name for the used basis gate set
- est_prim_str: str: Name that defines what estimator is used. Possible options are "aer" for the Aer Estimator, "terra" for the qiskit-terra Estimator, "ibm_runtime" for the IBM runtime Estimator or "ion_trap" for the Ion Trap Estimator (not yet implemented)
- backend_str: str: String that defines the used backend in the Estimator. For IBM runtime Estimator this string has to match the expected backend string, e.g., "ibmq_qasm_simulator" or "ibm_cairo". For Aer Estimator the string should be "AerSimulator" and for Terra Estimator the string should be "statevector_simulator".

The VQEEstimator class expects a EstimatorCalibration object as an input and a Sessions object if IBM runtime is used (otherwise this can be None).

Note that VQEEstimator has already the option to generate a Ion Trap estimator via est_prim_str == "ion_trap" but the corresponding if-part in the following functions raises a NotImplemented error and still needs to be implemented:
- _validate_estimator_options() of the EstimatorCalibration class
- get_filevector() of the EstimatorCalibration class
- _get_estimator() of the VQEEstimator class

### Optimizer Calibration

### Running the VQE


## Relavant qiskit links
[vqe-ibm-runtime-tutorial](https://qiskit.org/ecosystem/ibm-runtime/tutorials/vqe_with_estimator.html)
[Qiskit VQE classe](https://qiskit.org/ecosystem/algorithms/stubs/qiskit_algorithms.VQE.html)

