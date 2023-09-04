from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Iterable, Sequence
from . import Calibration as cal
import copy

class ResultData:
    def __init__(self,
                 energy: float,
                 **kwargs):
        self.energy = energy
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        string_list = []
        for key, val in self.to_dict().items():
            string_list.append(f"{key}={val}")
        out = "ResultData(%s)" % ", ".join(string_list)
        return out
    
    def to_dict(self):
        return copy.copy(self.__dict__)

    def get_filevector(self):
        """
        Define method to write the result data in a list format

        output: (header, data)
        """
        header = []
        data = []

        res_dict = self.to_dict()

        for key, val in res_dict.items():
            if isinstance(val, Sequence):
                for i in range(len(val)):
                    header.append(key+str(i))
                    data.append(val[i])
            else:
                header.append(key)
                data.append(val)

        return header, data

class ReferenceResult:
    def __init__(self,
                 data: ResultData,
                 cal_list: List[cal.Calibration]):
        self._data = data
        self._calibration_list = cal_list

    @property
    def calibration_list(self):
        return self._calibration_list

    @property
    def data(self):
        return self._data
    
    def __repr__(self):
        out = "ReferenceResult(data={}, cal_list={})".format(self._data, self._calibration_list)
        return out
    
    def to_dict(self):
        result_dict = self._data.to_dict()
        cnt = 0
        for cal in self._calibration_list:
            cal_dict = cal.to_dict()
            cal_name = cal_dict.pop("name", None)
            if cal_name is None:
                cal_name = "unknown_cal"+str(cnt)
                cnt += 1
            result_dict[cal_name] = cal_dict
        
        return result_dict

    def get_filevector(self):
        """
        Define method to write the summarized (shorted) result in a list format

        output: (header, data)
        """
        header = []
        data = []
        
        for cal in self._calibration_list:
            cal_header, cal_data = cal.get_filevector()
            header.extend(cal_header)
            data.extend(cal_data)

        curr_header, curr_data = self._data.get_filevector()
        header.extend(curr_header)
        data.extend(curr_data)
        
        return header, data

class VQEResult:
    def __init__(self,
                 vqe_data: ResultData,
                 vqe_cal_list: List[cal.Calibration],
                 reference_result: Union[ReferenceResult, None] = None) -> None:
        self._data = vqe_data
        self._calibration_list = vqe_cal_list
        self._reference = reference_result

    @property
    def calibration_list(self):
        return self._calibration_list

    @property
    def data(self):
        return self._data

    @property
    def reference(self):
        return self._reference

    def __repr__(self):
        out = "VQEResult(vqe_data={}, vqe_cal_list={}, reference_result={})".format(self._data, self._calibration_list, self._reference)
        return out

    def to_dict(self):
        result_dict = self._data.to_dict()
        cnt = 0
        for cal in self._calibration_list:
            cal_dict = cal.to_dict()
            cal_name = cal_dict.pop("name", None)
            if cal_name is None:
                cal_name = "unknown_cal"+str(cnt)
                cnt += 1
            result_dict[cal_name] = cal_dict

        if self._reference is None:
            result_dict["reference"] = None
        else:
            result_dict["reference"] = self._reference.to_dict()

        return result_dict

    def get_filevector(self):
        """
        Define method to write the summarized (shorted) result in a list format

        output: (header, data)
        """
        header = []
        data = []

        for cal in self._calibration_list:
            cal_header, cal_data = cal.get_filevector()
            header.extend(cal_header)
            data.extend(cal_data)

        if self._reference is not None:
            curr_header, curr_data = self._reference.data.get_filevector()
            curr_header = [s + "_ref" for s in curr_header]
            header.extend(curr_header)
            data.extend(curr_data)

            curr_header, curr_data = self._data.get_filevector()
            curr_header = [s + "_vqe" for s in curr_header]
            header.extend(curr_header)
            data.extend(curr_data)
        else:
            curr_header, curr_data = self._data.get_filevector()
            header.extend(curr_header)
            data.extend(curr_data)
        
        return header, data

class InferenceResult:
    def __init__(self,
                 inference_data: ResultData,
                 inference_cal_list: List[cal.Calibration],
                 vqe_result: VQEResult,
                 metadata: Dict) -> None:
        self._data = inference_data
        self._calibration_list = inference_cal_list
        self._vqe_reference = vqe_result
        self._metadata = metadata

    @property
    def calibration_list(self):
        return self._calibration_list

    @property
    def data(self):
        return self._data

    @property
    def vqe_reference(self):
        return self._vqe_reference

    @property
    def metadata(self):
        return self._metadata

    def __repr__(self):
        out = "InferenceResult(inference_data={}, inference_cal_list={}, vqe_result={}, metadata={})".format(self._data, self._calibration_list, self._vqe_reference, self._metadata)
        return out
    
    def to_dict(self):
        result_dict = self._data.to_dict()
        cnt = 0
        for cal in self._calibration_list:
            cal_dict = cal.to_dict()
            cal_name = cal_dict.pop("name", None)
            if cal_name is None:
                cal_name = "unknown_cal"+str(cnt)
                cnt += 1
            result_dict[cal_name] = cal_dict

        result_dict["vqe_reference"] = self._vqe_reference.to_dict()
        result_dict["metadata"] = self._metadata

        return result_dict

    def get_filevector(self):
        """
        Define method to write the summarized (shorted) result in a list format

        output: (header, data)
        """
        header = []
        data = []

        for cal in self._calibration_list:
            cal_header, cal_data = cal.get_filevector()
            header.extend(cal_header)
            data.extend(cal_data)

        vqe_ref_data_no_angles = copy.deepcopy(self._vqe_reference.data)
        #print(vqe_ref_data_no_angles)
        # remove double display of angles in filevector
        if hasattr(self._data, "angles"):
            delattr(vqe_ref_data_no_angles, "angles")
        curr_header, curr_data = vqe_ref_data_no_angles.get_filevector()
        curr_header = [s + "_vqe" for s in curr_header]
        header.extend(curr_header)
        data.extend(curr_data)

        curr_header, curr_data = self._data.get_filevector()
        curr_header = [s + "_infer" for s in curr_header]
        header.extend(curr_header)
        data.extend(curr_data)
        
        return header, data