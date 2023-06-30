from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
import copy

from . import VQEAnsatz as VQEA
from . import VQETargetModel as VQETM
from . import VQEOptimizer as VQEO
from . import VQEEstimator as VQEE
from . import VQEResult as VQER
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver, VQEResult
from qiskit.algorithms.algorithm_result.AlgorithmResult import MinimumEigensolverResult
from qiskit.quantum_info import Statevector
from qiskit import Aer, IBMQ, execute
from qiskit.primitives import BaseEstimator

def get_data_from_VQEResult(result: VQEResult,
                            opt_converged: bool,
                            overlap: Union[float, None] = None) -> VQER.ResultData:
    # Input data types
    # - result =  qiskit.algorithms.minimum_eigensolvers.VQEResult
    # - opt_converged True if optimization is converged
    result_out_dict = {}
    # total number of cost function calls
    tot_num_cost_fctn_calls = result.optimizer_result.nfev
    # number of optimizer iterations
    opt_evals = result.optimizer_result.nit
    # extract final energy
    E0_vqs = result.eigenvalue.real
    # extract optimal angles
    angles = result.optimal_parameters

    result_out = VQER.ResultData(E0_vqs)
    # check if auxilary operators were measured
    if result.aux_operators_evaluated:
        for key, val in result.aux_operators_evaluated.items():
            result_out_dict[key] = val[0].real

    result_out_dict["opt_converged"] = opt_converged
    result_out_dict["tot_num_cost_fctn_calls"] = tot_num_cost_fctn_calls
    result_out_dict["opt_iterations"] = opt_evals
    result_out_dict["overlap"] = overlap
    result_out_dict["angles"] = angles

    for key, val in result_out_dict.items():
        setattr(result_out, key, val)
    
    
    return result_out

def get_state_from_VQEResult(result: VQEResult) -> Statevector:
    # generate state via statevector simulator from optimal circuit angles
    backend_state = Aer.get_backend("statevector_simulator")
    # generate circuit with optimized angles
    circ_final = result.optimal_circuit.bind_parameters(angles)
    job = execute(circ_final, backend_state)
    psi_vqs = Statevector(job.result().get_statevector(circ_final))

    return psi_vqs
    

def get_data_from_MinimumEigensolverResult(result: MinimumEigensolverResult) -> VQER.ResultData:
    # Input data types
    # - result =  qiskit.algorithms.algorithm_result.AlgorithmResult.MinimumEigensolverResult
    # extract final energy
    E0_exact= result.eigenvalue.real
    result_out = VQER.ResultData(E0_exact)
    if result.aux_operators_evaluated:
        for key, val in result.aux_operators_evaluated.items():
            setattr(result_out, key, val[0].real)

        
    return result_out

def getOverlap(psi_vqs: Statevector,
               psi_ED: Statevector):
    # Input data types
    # - psi_vqs = qiskit.quantum_info.Statevector
    # - psi_ED = qiskit.quantum_info.Statevector
    
    return np.abs(psi_vqs.inner(psi_ED))

def run_exact_diagonalization(target_model: VQETM.VQETargetModel) -> Tuple[VQER.ReferenceResult, Statevector]:
    # generate hamiltonian with all penalties
    H = target_model.hamiltonian

    pen = target_model.get_ed_penalty()
    if pen is not None:
        H_p = H.add(pen)
    else:
        H_p = H

    # get dict with all additional observavbles
    aux_ops = target_model.aux_ops

    # exact diagonalization solver
    npme = NumPyMinimumEigensolver()

    # run ed
    result = npme.compute_minimum_eigenvalue(operator=H_p, aux_operators=aux_ops)

    # return data in correct format
    result_data = get_data_from_MinimumEigensolverResult(result)

    result_out = VQER.ReferenceResult(result_data, [target_model])
    # extract eigenstate
    psi_gs = Statevector(result.eigenstate)

    return result_out, psi_gs
    
def run_vqe(vqe_estimator: VQEE.VQEEstimator,
            target_model: VQETM.VQETargetModel,
            vqe_ansatz: VQEA.VQEAnsatz,
            vqe_optimizer: VQEO.VQEOptimizer,
            ref_result: Union[VQER.ReferenceResult, None] = None,
            ref_state: Union[Statevector, None] = None,
            save_iresults: bool = False,
            print_status: bool = False) -> Tuple[VQER.VQEResult, Statevector, Dict]:

    # store intermediate results via callback function
    iresults_dict = {}
    
    num_cost_fctn_calls = []
    energy_values = []
    circ_params = []
    est_meta = []
    def store_intermediate_cost_fctn_calls(eval_count, params, mean, meta):
        # number of cost function calls
        num_cost_fctn_calls.append(eval_count)
        # current cost value
        energy_values.append(mean)
        # current set of optimization parameters
        circ_params.append(params)
        # current estimator meta data
        est_meta.append(meta)

    # get estimator primitive
    estimator = vqe_estimator.estimator
    # print input data
    if print_status:
        print("Running VQE with")
        print("- estimator = ={}".format(estimator))
        print("- target model = {}".format(target_model))
        print("- vqe ansatz = {}". format(vqe_ansatz))
        print("- vqe optimizer = {}".format(vqe_optimizer))
        print("- save interm. results = {}".format(save_iresults))

    # generate hamiltonian with possible penalty
    H = target_model.hamiltonian
    pen = target_model.get_vqe_penalty()
    if pen is not None:
        H_p = H.add(pen)
    else:
        H_p = H

    # get dict with all additional observavbles
    aux_ops = target_model.aux_ops

    # get parametric quantum circuit
    circ = vqe_ansatz.circuit

    # get possible initial point
    param_init = vqe_optimizer.parameters.param_map_init

    # if no initial point was provided, use a random point
    if param_init is None:
        param_init = np.random.rand(circ.num_parameters)*2*np.pi
    if isinstance(param_init, Dict):
        # convert dict to list of values
        param_init = list(param_init.values())
        
    if isinstance(param_init, Sequence):
        # convert list to numpy array
        param_init = np.asarray(param_init)
        if param_init.size != circ.num_parameters:
            raise ValueError("number of initial parameters does not match number of circuit parameters!")

    callback_fctn = None
    if save_iresults:
        callback_fctn = store_intermediate_cost_fctn_calls

    # setup vqe object
    vqe = VQE(estimator, circ, vqe_optimizer.optimizer, gradient=vqe_optimizer.get_gradient(estimator), initial_point=param_init, callback=callback_fctn)

    # run vqe
    result = vqe.compute_minimum_eigenvalue(operator=H_p, aux_operators=aux_ops)
    opt_converged=True
    psi_vqe = get_state_from_VQEResult(result)
    if ref_state is None:
        overlap = None
    else:
        overlap = get_overlap(psi_vqe, ref_state)
    
    result_data = get_data_from_VQEResult(result, opt_converged, overlap = overlap)

    result_out = VQER.VQEResult(result_data, [target_model.parameters, vqe_ansatz.parameters, vqe_optimizer.parameters, vqe_estimator.parameters], reference_result = ref_result)

    if print_status:
        print("optimization finished:")
        result_data_dict = result_data.to_dict()
        for key, val in result_data_dict.items():
            print("- {} = {}".format(key, val))
        

    if save_iresults:
        iresults_dict["num_cost_fctn_calls"] = num_cost_fctn_calls
        iresults_dict["energy_values"] = energy_values
        iresults_dict["circ_params"] = circ_params
        iresults_dict["est_meta"] = est_meta
    

    return result_out, psi_vqe, iresults_dict

def get_iresults_filevector(iresults_dict: Dict) -> Tuple[List, List]:

    header = []
    data = []
    
    num_cost_fctn_calls = iresults_dict["num_cost_fctn_calls"]
    energy_values = iresults_dict["energy_values"]
    circ_params = iresults_dict["circ_params"]
    header = ["#num_cost_fctn_calls", "energy_values"]
    for n in range(len(circ_params[0])):
        header.append("circ_param"+str(n))

    for i in range(len(num_cost_fctn_calls)):
        curr_data = []
        curr_data.append(num_cost_fctn_calls[i])
        curr_data.append(energy_values[i])
        curr_circ_params = circ_params[i]
        for j in range(len(curr_circ_params)):
            curr_data.append(curr_circ_params[j])

        data.append(copy.copy(curr_data))

    return header, data
    
        
