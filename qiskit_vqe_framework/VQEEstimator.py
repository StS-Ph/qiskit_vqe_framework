from __future__ import annotations
from importlib.metadata import version
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from . import Calibration as cal
from qiskit.primitives import BaseEstimator
from qiskit.primitives import Estimator as TerraEstimator
from qiskit.primitives import BackendEstimator as BackendEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel

from qiskit.transpiler import PassManager

import qiskit_ibm_runtime as qir
import copy
import os
import yaml
import pickle

class EstimatorCalibration(cal.Calibration):
    def __init__(self,
                 est_opt: Dict,
                 noise_model_str: str,
                 coupling_map_str: str,
                 basis_gates_str: str,
                 est_prim_str: str,
                 backend_str: str) -> None:
        super().__init__("EstimatorCalibration")
        self._estimator_options = self._validate_estimator_options(est_opt, est_prim_str)
        self.noise_model_str = noise_model_str
        self.coupling_map_str = coupling_map_str
        self.basis_gates_str = basis_gates_str
        self._estimator_str = est_prim_str
        self.backend_str = backend_str

    @property
    def estimator_options(self):
        return self._estimator_options
    @estimator_options.setter
    def estimator_options(self,
                          est_opt: Dict):
        self._estimator_options = self._validate_estimator_options(est_opt, self.estimator_str)

    @property
    def estimator_str(self):
        return self._estimator_str

    def __repr__(self) -> str:
        out = "EstimatorCalibration(est_opt={}, noise_model_str={}, coupling_map_str={}, basis_gates_str={}, est_prim_str={}, backend_str={})".format(self._estimator_options, self.noise_model_str, self.coupling_map_str, self.basis_gates_str, self.estimator_str, self.backend_str)

        return out

    def to_dict(self):
        est_cal_dict = super().to_dict()
        est_str = est_cal_dict.pop("_estimator_str")
        est_cal_dict["estimator_str"] = est_str

        est_opt = est_cal_dict.pop("_estimator_options")
        est_cal_dict["estimator_options"] = est_opt

        return est_cal_dict

    def to_yaml(self,
                fname: str):
        # convert to dictionary
        est_cal_dict = self.to_dict()
        # search for noise_model (should not be contained in the yaml but pickled 
        for key in est_cal_dict["estimator_options"].keys():
            # check only the dictionaries in estimator options
            if isinstance(est_cal_dict["estimator_options"][key], Dict):
                # check if noise_model key exists
                noise_model = est_cal_dict["estimator_options"][key].pop("noise_model", None)
                # if noise_model is not None, replace with noise_model_str in yaml (when loaded this will be again replaced by pickled noise_model)
                if noise_model is not None:

                    fname_noise_model, yaml_ext = os.path.splitext(fname)
                    fname_noise_model = fname_noise_model + "_noise_model.pickle"
                    
                    if os.path.isfile(fname_noise_model):
                        raise ValueError("file for saving noise_model {} does already exist!".format(fname_noise_model))
                    with open(fname_noise_model, "wb") as f:
                        pickle.dump(noise_model, f)

                    # est_cal_dict["estimator_options"][key]["noise_model"] = est_cal_dict["noise_model_str"]
                    est_cal_dict["estimator_options"][key]["noise_model"] = fname_noise_model
                    
        # check if file already exists
        if os.path.isfile(fname):
            raise ValueError("file {} does already exist!".format(fname))

        # dump calibration dictionary into yaml file
        with open(fname, "w") as f:
            yaml.dump(est_cal_dict, f)
            
    def _validate_estimator_options(self,
                                    est_opt_in: Dict,
                                    est_prim_str: str) -> Dict:
        est_opt = copy.copy(est_opt_in)
        if est_prim_str == "aer":
            sub_cat = ["transpilation_options", "backend_options", "run_options", "approximation", "skip_transpilation", "abelian_grouping"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
            transp_opt = est_opt["transpilation_options"]
            if transp_opt is None:
                est_opt["transpilation_options"] = {}
            elif not isinstance(transp_opt, Dict):
                raise ValueError("transpilation options must be a dictionary!")

            backend_opt = est_opt["backend_options"]
            if backend_opt is None:
                est_opt["backend_options"] = {}
            elif not isinstance(backend_opt, Dict):
                raise ValueError("backend options must be a dictionary!")

            run_opt = est_opt["run_options"]
            if run_opt is None:
                est_opt["run_options"] = {}
            elif not isinstance(run_opt, Dict):
                raise ValueError("run options must be a dictionary!")

            circ_opt_lvl = est_opt["transpilation_options"].get("optimization_level", None)
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
                print("No circuit optimization level was chosen. Set it to default {}.".format(circ_opt_lvl))
                est_opt["transpilation_options"]["optimization_level"] = circ_opt_lvl
            if est_opt["skip_transpilation"] and circ_opt_lvl != 0:
                raise ValueError("skip transpilation flag was set to True. Can't do circuit optimization (level = {}) if transpilation is skipped.".format(circ_opt_lvl))

            if "shots" not in est_opt["run_options"].keys():
                if "shots" not in est_opt["backend_options"].keys():
                    shots = None
                else:
                    shots = est_opt["backend_options"].get("shots")
            else:
                shots = est_opt["run_options"].get("shots")

            if shots is None:
                if est_opt["approximation"] is False:
                    print("number of measurement shots is undefined and approximantion flag was set to False!")
                    
                    shots = 1024
                    print("set number of shots to {} instead of using backend default...".format(shots))
            if "shots" in est_opt["run_options"].keys():
                est_opt["run_options"]["shots"] = shots
            elif "shots" in est_opt["backend_options"].keys():
                est_opt["backend_options"]["shots"] = shots
            else:
                print("No shots key found. Add shots key to run options with value {}".format(shots))
                est_opt["run_options"]["shots"] = shots

            abelian_grouping = est_opt["abelian_grouping"]
            if not isinstance(abelian_grouping, bool):
                raise ValueError("Abelian grouping flag must be bool!")


            
        elif est_prim_str == "ibm_runtime":
            sub_cat = ["optimization_level", "resilience_level", "max_execution_time", "transpilation_options", "resilience_options", "execution_options", "environment_options", "simulator_options"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))

            err_mitig_meth = est_opt["resilience_level"]
            if err_mitig_meth is None:
                err_mitig_meth = 0
                print("error mitigation method is undefined. Set it to {} as default.".format(err_mitig_meth))
                est_opt["resilience_level"] = err_mitig_meth
            
            circ_opt_lvl = est_opt["optimization_level"]
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
                print("circuit optimization level is undefined. Set it to {} as default.".format(circ_opt_lvl))
                est_opt["optimization_level"] = circ_opt_lvl

            if not isinstance(est_opt["transpilation_options"], Dict):
                raise ValueError("transpilation options must be a dictionary")
            if not isinstance(est_opt["resilience_options"], Dict):
                raise ValueError("resilience options must be a dictionary")
            if not isinstance(est_opt["execution_options"], Dict):
                raise ValueError("execution options must be a dictionary")
            if not isinstance(est_opt["environment_options"], Dict):
                raise ValueError("environment options must be a dictionary")
            if not isinstance(est_opt["simulator_options"], Dict):
                raise ValueError("simulator options must be a dictionary")
            
            
            
            
            # if shots is not set, set it to default
            shots = est_opt["execution_options"].get("shots", None)
            if shots is None:
                shots = 1024
                print("number of shots is undefined. Set it to {} as default.".format(shots))
                est_opt["execution_options"]["shots"] = shots


        elif est_prim_str == "terra":
            sub_cat = ["run_options"]
            sub_cat.sort()

            est_opt_keys_sorted = sorted(list(est_opt.keys()))

            if est_opt_keys_sorted != sub_cat:
                raise ValueError("estimator options dictionaries sub-catagories {} do not match required sub-catagories based on estimator string {}!".format(est_opt_keys_sorted, sub_cat))
            if est_opt["run_options"] is None:
                est_opt["run_options"] = {}
            if not isinstance(est_opt["run_options"], Dict):
                raise ValueError("run options must be a dictionary!")        
        else:
            raise ValueError("estimator string {} does not match any known string!".format(est_prim_str))

        return est_opt
    
    def get_filevector(self) -> Tuple[List, List]:
        """
        Define method to write the summarized (shorted) calibration data in a list format

        output: (header, data)
        """

        header = []
        data = []

        header.append("estimator_str")
        data.append(self.estimator_str)

        err_mitig_meth = None
        circ_opt_lvl = None
        shots = None
        abelian_grouping = False

        if self.estimator_str == "aer":
            err_mitig_meth = 0
            circ_opt_lvl = self.estimator_options["transpilation_options"].get("optimization_level")
            shots = self.estimator_options["run_options"].get("shots", None)
            if shots is None:
                shots = self.estimator_options["backend_options"].get("shots", 0)
            abelian_grouping = self.estimator_options["abelian_grouping"]
                
        elif self.estimator_str == "ibm_runtime":
            err_mitig_meth = self.estimator_options["resilience_level"]

            circ_opt_lvl = self.estimator_options["optimization_level"]
            
            shots = self.estimator_options["execution_options"].get("shots")
            # IBM estimator always uses abelian_grouping 
            # https://quantumcomputing.stackexchange.com/questions/34694/is-qiskits-estimator-primitive-running-paulistrings-in-parallel
            abelian_grouping = True

        elif self.estimator_str == "terra":
            err_mitig_meth = 0
            circ_opt_lvl = 0
            shots = self.estimator_options["run_options"].get("shots", 0)

        elif self.estimator_str == "ion_trap":
            err_mitig_meth = 0
            circ_opt_lvl = self.estimator_options["transpilation_options"].get("optimization_level", None)
            if circ_opt_lvl is None:
                circ_opt_lvl = 0
            shots = self.estimator_options["run_options"].get("shots", 0)
            abelian_grouping = self.estimator_options["abelian_grouping"]
        
        header.append("err_mitigation")
        data.append(err_mitig_meth)

        header.append("circ_optimization")
        data.append(circ_opt_lvl)

        header.append("meas_shots")
        data.append(shots)
        
        header.append("backend_str")
        data.append(self.backend_str)
        
        header.append("noise_model")
        data.append(self.noise_model_str)
        
        header.append("coupling_map")
        data.append(self.coupling_map_str)

        header.append("basis_gates")
        data.append(self.basis_gates_str)

        header.append("abelian_grouping")
        data.append(abelian_grouping)
        

        return header, data
    
def get_EstimatorCalibration_from_dict(est_cal_dict: dict) -> EstimatorCalibration:

    est_opt = est_cal_dict.pop("estimator_options", None)
    if est_opt is None:
        raise ValueError("could not retrieve estimator options!")
    # check if noise_model is either None or a valid qiskit aer NoiseModel object
    noise_model_str = est_cal_dict.pop("noise_model_str", None)
    if noise_model_str is None:
        raise ValueError("could not retrieve noise model string from file!")
    
    valid_noise_model_found = False
    for key in est_opt.keys():
        if isinstance(est_opt[key], Dict):
            noise_model = est_opt[key].get("noise_model", None)
            if noise_model is not None:
                if not isinstance(noise_model, NoiseModel):
                    raise ValueError("Loaded noise model is no qiskit_aer NoiseModel!")
                if noise_model_str == "None":
                    raise ValueError("Noise model is a valid aer NoiseModel but noise model string equals string None!")
                valid_noise_model_found = True

    if not valid_noise_model_found:
        if noise_model_str != "None":
            raise ValueError("Noise model is None but noise model string {} does not match string None.".format(noise_model_str))
        
    coupling_map_str = est_cal_dict.pop("coupling_map_str", None)
    if coupling_map_str is None:
        raise ValueError("could not retrieve coupling map string from file!")
    basis_gates_str = est_cal_dict.pop("basis_gates_str", None)
    if basis_gates_str is None:
        raise ValueError("could not retrieve basis gates string from file!")
    
    est_prim_str = est_cal_dict.pop("estimator_str", None)
    if est_prim_str is None:
        raise ValueError("could not retrieve estimator string from file!")
    backend_str = est_cal_dict.pop("backend_str", None)
    if backend_str is None:
        raise ValueError("could not retrieve backend string from file!")

    name = est_cal_dict.pop("name", None)

    est_cal = EstimatorCalibration(est_opt, noise_model_str, coupling_map_str, basis_gates_str, est_prim_str, backend_str)
    return est_cal

def get_EstimatorCalibration_from_yaml(fname: str) -> EstimatorCalibration:
    
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    est_cal_dict = None
    raw_data = None
    with open(fname, "r") as f:
        raw_data = f.read()

    est_cal_dict = yaml.load(raw_data, Loader=yaml.Loader)
    if est_cal_dict is None:
        raise ValueError("Something went wrong while reading in yml text file! resulting dictionary is empty!")
    
    est_opt = est_cal_dict.get("estimator_options", None)
    if est_opt is None:
        raise ValueError("could not retrieve estimator options!")
    
    # load possible noise_model
    for key in est_opt.keys():
        if isinstance(est_opt[key], Dict):
            fname_noise_model = est_opt[key].get("noise_model", None)
            if fname_noise_model is not None:
                
                if not os.path.isfile(fname_noise_model):
                    raise ValueError("Unable to find pickle file to load noise_model for estimator option {}. Looked for file {}.".format(key, fname_noise_model))
                
                with open(fname_noise_model, "rb") as f:
                    noise_model = pickle.load(f)

                if not isinstance(noise_model, NoiseModel):
                    raise ValueError("Loaded noise model is no qiskit_aer NoiseModel!")
                
                est_cal_dict["estimator_options"][key]["noise_model"] = noise_model

    return get_EstimatorCalibration_from_dict(est_cal_dict)
    

def get_EstimatorCalibration_from_pickle(fname: str) -> EstimatorCalibration:
    if not os.path.isfile(fname):
        raise ValueError("file {} does not exist!".format(fname))

    est_cal = None
    with open(fname, "rb") as f:
        est_cal = pickle.load(f)

    if not isinstance(est_cal, EstimatorCalibration):
        raise ValueError("loaded pickle object is no EstimatorCalibration!")

    return est_cal


class VQEEstimator:
    def __init__(self,
                 estimator_parameters: EstimatorCalibration,
                 session: Union[qir.Session, None] = None) -> None:
        self._parameters = estimator_parameters
        if self._parameters.estimator_str == "ibm_runtime":
            if session is None:
                raise ValueError("session must be a runtime session for ibm runtime estimator!")
        self._session = session
        self._estimator = self._get_estimator()

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self,
                   new_parameters: EstimatorCalibration):
        self._parameters = new_parameters
        self._update_estimator()
    @property
    def session(self):
        return self._session
    @session.setter
    def session(self,
                ses: Union[qir.Session, None]):
        if self._parameters.estimator_str == "ibm_runtime":
            if ses is None:
                raise ValueError("session must be a runtime session for ibm runtime estimator!")
        self._session = ses

    @property
    def estimator(self):
        return self._estimator

    def __repr__(self):
        out = "VQEEstimator(estimator_parameters={}, session={})".format(self.parameters, self.session)

        return out

    def to_dict(self):
        estimator_dict = {}
        estimator_dict["parameters"] = self.parameters.to_dict()
        estimator_dict["session"] = self.session
        estimator_dict["estimator"] = self.estimator

        return estimator_dict

    def update_parameters(self,
                          new_parameters: EstimatorCalibration) -> None:
        self.parameters = new_parameters

    def _update_estimator(self) -> None:
        self._estimator = self._get_estimator()

    def _get_estimator(self) -> BaseEstimator:
        options_dict = self._parameters.estimator_options
        if self._parameters.estimator_str == "aer":
            est = AerEstimator(backend_options=options_dict["backend_options"], transpile_options=options_dict["transpilation_options"], run_options=options_dict["run_options"], approximation=options_dict["approximation"], skip_transpilation=options_dict["skip_transpilation"], abelian_grouping=options_dict["abelian_grouping"])
        elif self._parameters.estimator_str == "ibm_runtime":
            options = qir.options.Options(optimization_level=options_dict["optimization_level"], resilience_level=options_dict["resilience_level"], max_execution_time=options_dict["max_execution_time"], transpilation=options_dict["transpilation_options"], resilience=options_dict["resilience_options"], execution=options_dict["execution_options"], environment=options_dict["environment_options"], simulator=options_dict["simulator_options"])
            est = qir.Estimator(session=self._session, options=options)
        elif self._parameters.estimator_str == "terra":
            est = TerraEstimator(options=options_dict["run_options"])
        else:
            raise ValueError("estimator string {} in parameters does not match any known string!".format(self._parameters.estimator_str))
        return est
        
                
                
                
                
