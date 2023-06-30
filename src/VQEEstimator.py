from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from . import Calibration as cal
from qiskit.primitives import BaseEstimator
from qiskit.primitives import Estimator as TerraEstimator
from qiskit_aer.primitives import Estimator as AerEstimator



import qiskit_ibm_runtime as qir
import copy

class EstimatorCalibration(cal.Calibration):
    def __init__(self,
                 est_opt: Dict,
                 noise_model_str: str,
                 coupling_map_str: str,
                 est_prim_str: str,
                 backend_str: str) -> None:
        super().__init__("EstimatorCalibration")
        self._estimator_options = self._validate_estimator_options(est_opt, est_prim_str)
        self.noise_model_str = noise_model_str
        self.coupling_map_str = coupling_map_str
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
        out = "EstimatorCalibration(est_opt={}, noise_model_str={}, coupling_map_str={}, est_prim_str={}, backend_str={})".format(self._estimator_options, self.noise_model_str, self.coupling_map_str, self.estimator_str, self.backend_str)

        return out

    def to_dict(self):
        est_cal_dict = super().to_dict()
        est_str = est_cal_dict.pop("_estimator_str")
        est_cal_dict["estimator_str"] = est_str

        est_opt = est_cal_dict.pop("_estimator_options")
        est_cal_dict["estimator_options"] = est_opt

        return est_cal_dict
    

    def _validate_estimator_options(self,
                                    est_opt_in: Dict,
                                    est_prim_str: str) -> Dict:
        est_opt = copy.copy(est_opt_in)
        if est_prim_str == "aer":
            sub_cat = ["transpilation_options", "backend_options", "run_options", "approximation", "skip_transpilation"]
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


            
        elif est_prim_str == "ibm_runtime":
            sub_cat = ["optimization_level", "resilience_level", "max_execution_time", "transpilation_options", "resilience_options", "execution_options", "enviroment_options", "simulator_options"]
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
            if not isinstance(est_opt["enviroment_options"], Dict):
                raise ValueError("enviroment options must be a dictionary")
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
            
        elif est_prim_str == "ion_trap":
            raise NotImplementedError
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

        if self.estimator_str == "aer":
            err_mitig_meth = 0
            circ_opt_lvl = self.estimator_options["transpilation_options"].get("optimization_level")
            shots = self.estimator_options["run_options"].get("shots", None)
            if shots is None:
                shots = self.estimator_options["backend_options"].get("shots", 0)
                
        elif self.estimator_str == "ibm_runtime":
            err_mitig_meth = self.estimator_options["resilience_level"]

            circ_opt_lvl = self.estimator_options["optimization_level"]
            
            shots = self.estimator_options["execution_options"].get("shots")

        elif self.estimator_str == "terra":
            err_mitig_meth = 0
            circ_opt_lvl = 0
            shots = self.estimator_options["run_options"].get("shots", 0)

        elif self.estimator_str == "ion_trap":
            raise NotImplementedError
        
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

        return header, data

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
            est = AerEstimator(backend_options=options_dict["backend_options"], transpile_options=options_dict["transpilation_options"], run_options=options_dict["run_options"], approximation=options_dict["approximation"], skip_transpilation=options_dict["skip_transpilation"])
        elif self._parameters.estimator_str == "ibm_runtime":
            options = qir.options.Options(optimization_level=options_dict["optimization_level"], resilience_level=options_dict["resilience_level"], max_execution_time=options_dict["max_execution_time"], transpilation=options_dict["transpilation_options"], resilience=options_dict["resilience_options"], execution=options_dict["execution_options"], enviroment=options_dict["enviroment_options"], simulator=options_dict["simulator_options"])
            est = qir.Estimator(session=self._session, options=options)
        elif self._parameters.estimator_str == "terra":
            est = TerraEstimator(options=options_dict["run_options"])
        elif self._parameters.estimator_str == "ion_trap":
            raise NotImplementedError
        else:
            raise ValueError("estimator string {} in parameters does not match any known string!".format(self._parameters.estimator_str))
        return est
        
                
                
                
                
