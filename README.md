# Qiskit VQE data framework

A unpublished project to provide a data framework for running vqe simulations in qiskit.
It provides a calibration class for each major part in a qiskit vqe simualtion (target model, circuit ansatz, optimizer and estimator). Which can be used to easily save all calibration data in seperate files as dictionaries or combined as a filevector in one file.

Each vqe part has additionally a class which contains its calibration class and the corresponding qiskit object. The qiskit objects are generated via methods of this class taking the calibration class as an input. This way calibration and corresponding qiskit object are correctly associated with each other.

Each Calibration class can export its data to a dictionary via the `to_dict()` function or to a .yaml file via the `to_yaml()` function. Additionally one can generate a list of the calibration data via the `get_filevector()` function. Each calibration class can be either generated from a dictionary via `get_*Calibration Class name*_from_dict(d: dict)` (inverse of `to_dict()` function) or from a yaml file via `get_*Calibration Class name*_from_yaml(filename: str)` or from the pickled Calibration Class object via `get_*Calibration Class name*_from_pickle(filename: str)`.

## Installation

In the current version of this package a lot of changes are made and it is not yet published. Thus it must be installed in editable mode via:
`pip install -e `
in the directory with the setup.py.

## Usage

### Ansatz Calibration

To calibrate a VQE ansatz circuit the Calibration class `AnsatzCalibration` (in VQEAnsatz.py) expects 4 input variables: 
- `num_qubits: int`: number of qubits in the circuit 
- `num_layers: int`: number of algorithm layers 
- `ansatz_str: str`: string that names the used ansatz 
- `psi_start: Union[str, list, None] = None`: the initial state vector 
Other attributes can also be provided based on what the particular ansatz expects via `attribute = attr_value`.

Note that every VQE ansatz needs to be implemented as a derived class of the `VQEAnsatz` class. Here additional calibration parameters can be handled appropriately in `__init__`. In the `__init__` function a `AnsatzCalibration` object must be generated which is then given to the parent `__init__` as an input. The derived class needs to implement a `_get_circuit()` function which generates the qiskit `QuantumCircuit` object corresponding to the Ansatz from the calibration data internally during the initialization.

### Target Model Calibration

To calibrate the VQE Target Model the calibration class `ModelCalibration` ( in VQETargetModel.py) only expects 1 input variable, i.e., a certain model name model_name: str. Since all additional parameters depend on the particular model. They can of course also be provided as `attribute = attr_value`.

Note that every VQE target model needs to be implemented as a derived class of the `VQETargetModel` class. Here all required calibration parameters can be handled appropriately in `__init__`. In the `__init__` function a `ModelCalibration` object must be generated which is then given to the parent `__init__` as an input. The derived class needs to implement the following abstract methods from `VQETargetModel`:
1. `_validate_parameters(model_parameters: ModelCalibration) -> None`: Define method to validate model parameters (raise ValueErrors if not valid)
2. `_get_hamiltonian(self) -> Union[PauliSumOp, SparsePauliOp]`: Define method to generate hamiltonian from model parameters
3. `_get_aux_ops(self) -> Union[Dict[str, Union[PauliSumOp, SparsePauliOp]], None]`: Define method to generate all observables that should be additionally measured (to Hamiltonian) in dict. format `{"name": observable}`. Return `None` if no additional observables should be measured.
4. `get_ed_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]`: Define method to generate energy penalty term in exact diagonalization. If no penalty, then return `None`
5. `get_vqe_penalty(self) -> Union[PauliSumOp, SparsePauliOp, None]`: Define method to generate energy penalty term in vqs. If no penalty, then return `None`

Methods 1-3 are then used in the parent `__init__` to generate the Hamiltonian and the auxillary observables from the calibration data internally during the initialization.

### Estimator Calibration

To calibrate the VQE Estimator, the calibration class `EstimatorCalibration` (in VQEEstimator.py) expects 6 input variables: 
- `est_opt: Dict`: All Options to calibrate the used estimator class
- `noise_model_str: str`: Unique name for the used noise model
- `coupling_map_str: str`: Unique name for the used coupling map
- `basis_gates_str: str`: Unique name for the used basis gate set
- `est_prim_str: str`: Name that defines what estimator is used. Possible options are `"aer"` for the Aer Estimator, `"terra"` for the qiskit-terra Estimator, `"ibm_runtime"` for the IBM runtime Estimator or `"ion_trap"` for the Ion Trap Estimator (not yet implemented)
- `backend_str: str`: String that defines the used backend in the Estimator. For IBM runtime Estimator this string determines the used backend! For example `"ibmq_qasm_simulator"` sets a simulation on the ibm qasm simulator or `"ibm_cairo"` sets a real hardware run on this device. For Aer Estimator the string should be `"AerSimulator"` and for Terra Estimator the string should be `"statevector_simulator"`, but for both this variable changes nothing in the simulation.

The `VQEEstimator` class expects a `EstimatorCalibration` object and a qiskit runtime `Session` object if IBM runtime is used (otherwise this can be `None`) as an input. The corresponding qiskit Estimator class is then generated via `_get_estimator()` internally from the calibration data during initialization.

Note that `VQEEstimator` has already the option to generate a Ion Trap estimator via est_prim_str == "ion_trap" but the corresponding if-part in the following functions raises a `NotImplemented` error and still needs to be implemented:
- `_validate_estimator_options()` of the `EstimatorCalibration` class
- `get_filevector()` of the `EstimatorCalibration` class
- `_get_estimator()` of the `VQEEstimator` class

### Optimizer Calibration

To calibrate the VQE Optimizer, the calibration class `OptimizerCalibration` (in VQEOptimizer.py) expects 5 input variables:
- `name: str`: Defines which optimizer should be used. Currently only `"SPSA"` is supported.
- `maxiter: int`: Maximum number of optimization iterations.
- `grad_meth: str`: String that should define what method to calculate the gradient is used in gradient-based optimization. Currently, since only `"SPSA"` is supported the string must be `"fin_diff"` (finite difference), otherwise a `ValueError` is raised.
- `param_map_init: Union[Sequence[float], Dict[str, float], None] = None`: Initial parameter vector. Usually this is chosen randomly and if thats the case it can be set to `None`.
- `termination_checker: Union[tc.TerminationChecker, None] = None`: `TerminationChecker` object (in TerminationChecker.py) that defines what method is used to calculate if the optimization is already converged (before `maxiter` is reached). Currently implemented options are checking the relative change in the previous energy values (`RelativeEnergyChecker`) or fitting a line to the previous energy values and checking its slope (`LinearFitChecker`). If this is set to `None` the optimization always run until it reaches `maxiter`.

The `VQEOptimizer` class expects a `OptimizerCalibration` object as an input. The corresponding qiskit optimizer object is then generated from this calibration datat internally via the `_get_optimizer()` function.
Any additional optimization methods need to be implemented in this function properly.

The `get_gradient()` function is currently not implemented and always return `None`.

### Running the VQE

In VQErun.py all function are collected, which are needed to run the VQE, a exact diagonalization (ED) or an inference run of a optimial vqe solution.

The `run_exact_diagonalization` function expects a `VQETargetModel` object as an input and run a ED for the corresponding Hamiltonian using the `NumPyMinimumEigensolver` class from qiskit. It returns the result data in form of a `VQEReferenceResult` object (see [next section](#Result-data)) and the ground state as a qiskit `Statevector` object.

The `run_vqe` function expects a `VQEEstimator` object, a `VQETargetModel` object, a `VQEAnsatz` object, and a `VQEOptimizer` object as an input. Additionally a reference result (e.g. ED result) `ref_result: ReferenceResult` and a reference ground state `ref_state: Statevector` can be given as an optional input to calculate the state overlap and to be associated with the vqe result in the `VQEResult` object. Intermediate results can be optionally stored via the `save_iresult: bool` flag in the input options. The current status of the vqe can be optionally printed to REPL via the `print_status: bool` flag in the input options. The function returns the result data as a `VQEResult` object, the approximate ground state as a `Statevector` object and the stored intermediate results as a dictionary (empty if no results are stored).

The `inference_run` function expects a `VQEEstimator` object, a `VQETargetModel` object, a `VQEAnsatz` object, and a `VQEResult` object as an input. It purpose is to re-run the optimal circuit from a VQE result on a real hardware device. The function returns the result data as a `InferenceResult` object.



### Result data

In order to handle the all result data in a unified way a general `ResultData` object, a `ReferenceResult` object, a `VQEResult` object and a `InferenceResult` object are defined in VQEResult.py.

The `ResultData` class expects a energy value as an input. All additional data can also be assigned to an attribute of the `ResultData` object via `attribute = attr_val`. A `ResultData` object can be converted to a dictionary via the `to_dict()` function or to a data vector via the `get_filevector()` function.

The `ReferenceResult` class expects a `ResultData` object and a list of `Calibration` objects (parent class of `ModelCalibration`, `AnsatzCalibration`, `EstimatorCalibration`, `OptimizerCalibration`) as an input. This class is intended to be used for result data that serves as a reference for a vqe result. It can be a ED result but also another vqe result.
The list of `Calibration` objects holds the information on how and from where the result data has been obtained. The `ReferenceResult` can be converted to a dictionary via the `to_dict()` function or to a data vector via the `get_filevector()` function.

The `VQEResult` class expects a `ResultData` object and a list of `Calibration` objects (parent class of `ModelCalibration`, `AnsatzCalibration`, `EstimatorCalibration`, `OptimizerCalibration`) as an input. This class is intended to be used for a vqe result data. Optionally a `ReferenceResult` object can be given as an input and thus be a assigned reference to the VQE result. The list of `Calibration` objects holds the information on how and from where the result data has been obtained. The `VQEResult` can be converted to a dictionary via the `to_dict()` function or to a data vector via the `get_filevector()` function.

The `InferenceResult` class expects a `ResultData` object, a list of `Calibration` objects (parent class of `ModelCalibration`, `AnsatzCalibration`, `EstimatorCalibration`, `OptimizerCalibration`), a `VQEResult` object and a metadata dictionary as an input. This class is intended to be used for a inference run of a optimial vqe solution on a real quantum hardware. The metadata dictionary should carry the metadata of all estimator results (energy and aux. observables). The list of `Calibration` objects holds the information on how and from where the result data has been obtained. The `InferenceResult` can be converted to a dictionary via the `to_dict()` function or to a data vector via the `get_filevector()` function.

## Relavant qiskit links

[vqe-ibm-runtime-tutorial](https://qiskit.org/ecosystem/ibm-runtime/tutorials/vqe_with_estimator.html)
[Qiskit VQE classe](https://qiskit.org/ecosystem/algorithms/stubs/qiskit_algorithms.VQE.html)

