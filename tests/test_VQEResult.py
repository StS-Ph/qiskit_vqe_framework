import unittest
import qiskit_vqe_framework
import qiskit_vqe_framework.VQEResult as VQER
import qiskit_vqe_framework.VQETargetModel as VQETM
import qiskit_vqe_framework.VQEAnsatz as VQEA
import qiskit_vqe_framework.VQEOptimizer as VQEO
import qiskit_vqe_framework.TerminationChecker as tc
import qiskit_vqe_framework.VQEEstimator as VQEE
import copy

class TestVQEResultData(unittest.TestCase):
    def setUp(self):
        self.result_data = VQER.ResultData(-1.0, N0 = 1.0, N1 = 0.0, N2=0.0, Qtot = 0.0, opt_converged = True, tot_num_cost_fctn_calls = 200, opt_iterations=100, overlap=0.98, angles = [0.0, 0.0, 0.0])

    def test_repr(self):
        self.assertEqual(repr(self.result_data), "ResultData(energy=-1.0, N0=1.0, N1=0.0, N2=0.0, Qtot=0.0, opt_converged=True, tot_num_cost_fctn_calls=200, opt_iterations=100, overlap=0.98, angles=[0.0, 0.0, 0.0])")

    def test_to_dict(self):
        self.assertEqual(self.result_data.to_dict(), {"energy": -1.0, "N0": 1.0, "N1": 0.0, "N2": 0.0, "Qtot": 0.0, "opt_converged": True, "tot_num_cost_fctn_calls": 200, "opt_iterations": 100, "overlap": 0.98, "angles": [0.0, 0.0, 0.0]})

    def test_get_filevector(self):
        header, data = self.result_data.get_filevector()

        self.assertEqual(header, ["energy", "N0", "N1", "N2", "Qtot", "opt_converged", "tot_num_cost_fctn_calls", "opt_iterations", "overlap", "angles0", "angles1", "angles2"])
        self.assertEqual(data, [-1.0, 1.0, 0.0, 0.0, 0.0, True, 200, 100, 0.98, 0.0, 0.0, 0.0])

class TestVQEReferenceResult(unittest.TestCase):
    def setUp(self):
        self.result_data = VQER.ResultData(-1.0, N0 = 1.0, N1 = 0.0, N2=0.0, Qtot = 0.0)
        self.model_cal = VQETM.ModelCalibration("transverse_field_Ising_model", num_spins=4, J=1.0, g=-0.5)
        self.ref_result = VQER.ReferenceResult(self.result_data, [self.model_cal])

    def test_repr(self):
        self.assertEqual(repr(self.ref_result), "ReferenceResult(data={}, cal_list={})".format(self.result_data, [self.model_cal]))

    def test_to_dict(self):
        model_cal_dict = self.model_cal.to_dict()
        name = model_cal_dict.pop("name")
        self.assertEqual(self.ref_result.to_dict(), {"energy": -1.0, "N0": 1.0, "N1": 0.0, "N2": 0.0, "Qtot": 0.0, "ModelCalibration": model_cal_dict})

    def test_get_filevector(self):
        header, data = self.ref_result.get_filevector()

        header_cal, data_cal = self.model_cal.get_filevector()
        header_data, data_data = self.result_data.get_filevector()

        header_cal.extend(header_data)
        data_cal.extend(data_data)
        
        self.assertEqual(header, header_cal)
        self.assertEqual(data, data_cal)

        print(header)
        print(data)
        
class TestVQEResult(unittest.TestCase):
    def setUp(self):
        self.vqe_data = VQER.ResultData(-1.0, N0 = 1.0, N1 = 0.0, N2=0.0, Qtot = 0.0, opt_converged = True, tot_num_cost_fctn_calls = 200, opt_iterations=100, overlap=0.98, angles = [0.0, 0.0, 0.0])
        self.model_cal = VQETM.ModelCalibration("transverse_field_Ising_model", num_spins=4, J=1.0, g=-0.5)
        self.ref_data = VQER.ResultData(-1.0, N0 = 1.0, N1 = 0.0, N2=0.0, Qtot = 0.0)
        self.ref_result = VQER.ReferenceResult(self.ref_data, [self.model_cal])
        self.ansatz_cal = VQEA.AnsatzCalibration(4, 1, "ESU2", [1.0,0.0,0.0,0.0])
        self.checker = tc.RelativeEnergyChecker(100, 20, 0.01)
        self.opt_cal = VQEO.OptimizerCalibration("SPSA", 100, "fin_diff", param_map_init = [0.0, 0.0, 0.0], termination_checker = self.checker)
        self.est_opt = {"transpilation_options": {"optimization_level": 3}, "backend_options": {"method": "automatic", "shots": 4000}, "run_options": {"shots": 1024}, "approximation": False, "skip_transpilation": False}
        self.estimator_cal = VQEE.EstimatorCalibration(self.est_opt, "None", "None", "aer", "aer_automatic")

        self.vqe_result = VQER.VQEResult(self.vqe_data, [self.model_cal, self.ansatz_cal, self.opt_cal, self.estimator_cal], reference_result=self.ref_result)

    def test_repr(self):
        self.assertEqual(repr(self.vqe_result), "VQEResult(vqe_data={}, vqe_cal_list={}, reference_result={})".format(self.vqe_data, [self.model_cal, self.ansatz_cal, self.opt_cal, self.estimator_cal], self.ref_result))

    def test_to_dict1(self):
        model_cal_dict = self.model_cal.to_dict()
        name = model_cal_dict.pop("name")
        ansatz_cal_dict = self.ansatz_cal.to_dict()
        name = ansatz_cal_dict.pop("name")
        opt_cal_dict = self.opt_cal.to_dict()
        name = opt_cal_dict.pop("name")
        estimator_cal_dict = self.estimator_cal.to_dict()
        name = estimator_cal_dict.pop("name")

        self.assertEqual(self.vqe_result.to_dict(), {"energy": -1.0, "N0": 1.0, "N1": 0.0, "N2": 0.0, "Qtot": 0.0, "opt_converged": True, "tot_num_cost_fctn_calls": 200, "opt_iterations": 100, "overlap": 0.98, "angles": [0.0, 0.0, 0.0] , "ModelCalibration": model_cal_dict, "AnsatzCalibration": ansatz_cal_dict, "OptimizerCalibration": opt_cal_dict, "EstimatorCalibration": estimator_cal_dict, "reference": self.ref_result.to_dict()})

    def test_to_dict2(self):
        self.vqe_result._reference = None
        model_cal_dict = self.model_cal.to_dict()
        name = model_cal_dict.pop("name")
        ansatz_cal_dict = self.ansatz_cal.to_dict()
        name = ansatz_cal_dict.pop("name")
        opt_cal_dict = self.opt_cal.to_dict()
        name = opt_cal_dict.pop("name")
        estimator_cal_dict = self.estimator_cal.to_dict()
        name = estimator_cal_dict.pop("name")

        self.assertEqual(self.vqe_result.to_dict(), {"energy": -1.0, "N0": 1.0, "N1": 0.0, "N2": 0.0, "Qtot": 0.0, "opt_converged": True, "tot_num_cost_fctn_calls": 200, "opt_iterations": 100, "overlap": 0.98, "angles": [0.0, 0.0, 0.0] , "ModelCalibration": model_cal_dict, "AnsatzCalibration": ansatz_cal_dict, "OptimizerCalibration": opt_cal_dict, "EstimatorCalibration": estimator_cal_dict, "reference": None})

    def test_get_filevector1(self):
        header = []
        data = []

        header_cal, data_cal = self.model_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.ansatz_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.opt_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.estimator_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_ref, data_ref = self.ref_data.get_filevector()
        header_ref = [s+"_ref" for s in header_ref]
        header.extend(header_ref)
        data.extend(data_ref)

        header_vqe, data_vqe = self.vqe_data.get_filevector()
        header_vqe = [s+"_vqe" for s in header_vqe]
        header.extend(header_vqe)
        data.extend(data_vqe)

        header_vqe_result, data_vqe_result = self.vqe_result.get_filevector()

        print(header_vqe_result)
        print(data_vqe_result)
        
        self.assertEqual(header_vqe_result, header)
        self.assertEqual(data_vqe_result, data)

    def test_get_filevector2(self):
        self.vqe_result._reference=None
        header = []
        data = []

        header_cal, data_cal = self.model_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.ansatz_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.opt_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        header_cal, data_cal = self.estimator_cal.get_filevector()
        header.extend(header_cal)
        data.extend(data_cal)

        
        header_vqe, data_vqe = self.vqe_data.get_filevector()
        header.extend(header_vqe)
        data.extend(data_vqe)

        header_vqe_result, data_vqe_result = self.vqe_result.get_filevector()

        print(header_vqe_result)
        print(data_vqe_result)
        
        self.assertEqual(header_vqe_result, header)
        self.assertEqual(data_vqe_result, data)


