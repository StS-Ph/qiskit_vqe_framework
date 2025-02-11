import unittest
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit import IBMQ
import qiskit_vqe_framework
import qiskit_vqe_framework.VQEEstimator as VQEE
import os

class TestVQEEstimatorCalibration(unittest.TestCase):
    def setUp(self):
        self.est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": 3}, "backend_options": {"method": "automatic", "shots": 4000}, "run_options": {"shots": 1024}, "approximation": False, "skip_transpilation": False}
        self.estimator_cal = VQEE.EstimatorCalibration(self.est_opt, "None", "None", "None", "aer", "aer_automatic")

    def test_repr(self):
        self.assertEqual(repr(self.estimator_cal), "EstimatorCalibration(est_opt={}, noise_model_str=None, coupling_map_str=None, basis_gates_str=None, est_prim_str=aer, backend_str=aer_automatic)".format(self.est_opt))

    def test_to_dict(self):
        self.assertEqual(self.estimator_cal.to_dict(), {"name": "EstimatorCalibration", "noise_model_str": "None", "coupling_map_str": "None", "basis_gates_str": "None", "backend_str": "aer_automatic", "estimator_str": "aer", "estimator_options": self.est_opt})

    def test_get_filevector(self):
        header, data = self.estimator_cal.get_filevector()

        self.assertEqual(set(header), set(["estimator_str", "err_mitigation", "circ_optimization", "meas_shots", "backend_str", "noise_model", "coupling_map", "basis_gates", "abelian_grouping"]))
        self.assertEqual(set(data), set(["aer", 0, 3, 1024, "aer_automatic", "None", "None", "None", False]))
    def test_validate_estimator_options_aer(self):
        est_prim_str = "aer"
        est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": 3}, "backend_options": {"method": "automatic", "shots": 4000}, "approximation": False, "skip_transpilation": False, "run_options": {"shots": 1024}}
        self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)

        est_opt = {"run_options": None}
        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"abelian_grouping": False, "transpilation_options": None, "backend_options": None, "approximation": False, "skip_transpilation": False, "run_options": None}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)
        self.assertEqual(val_est_opt["transpilation_options"], {"optimization_level": 0})
        self.assertEqual(val_est_opt["backend_options"], {})
        self.assertEqual(val_est_opt["run_options"], {"shots": 1024})

        est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": None}, "backend_options": {"method": "automatic", "shots": 4000}, "approximation": False, "skip_transpilation": False, "run_options": {"shots": 1024}}
        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)
        self.assertEqual(val_est_opt["transpilation_options"]["optimization_level"], 0)
        est_opt["skip_transpilation"] = True
        est_opt["transpilation_options"]["optimization_level"] = 1

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": None}, "backend_options": {"method": "automatic", "shots": None}, "approximation": False, "skip_transpilation": False, "run_options": {}}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)
        self.assertEqual(val_est_opt["backend_options"]["shots"], 1024)

        est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": None}, "backend_options": {"method": "automatic"}, "approximation": False, "skip_transpilation": False, "run_options": {"shots": None}}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)
        self.assertEqual(val_est_opt["run_options"]["shots"], 1024)

        est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": None}, "backend_options": {"method": "automatic"}, "approximation": False, "skip_transpilation": False, "run_options": {}}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)
        self.assertEqual(val_est_opt["run_options"]["shots"], 1024)

        est_opt = {"abelian_grouping": "True", "transpilation_options": {"optimization_level": 3}, "backend_options": {"method": "automatic", "shots": 4000}, "run_options": {"shots": 1024}, "approximation": False, "skip_transpilation": False}
        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

    def test_validate_estimator_options_ibm_runtime(self):
        est_prim_str = "ibm_runtime"
        est_opt = {"run_options": None}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {}, "environment_options": {}, "simulator_options": {}}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)

        self.assertEqual(val_est_opt["resilience_level"], 0)
        self.assertEqual(val_est_opt["optimization_level"], 0)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": None, "resilience_options": {}, "execution_options": {}, "environment_options": {}, "simulator_options": {}}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": None, "execution_options": {}, "environment_options": {}, "simulator_options": {}}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": None, "environment_options": {}, "simulator_options": {}}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {}, "environment_options": None, "simulator_options": {}}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {}, "environment_options": {}, "simulator_options": None}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"optimization_level": None, "resilience_level": None, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {}, "environment_options": {}, "simulator_options": {}}

        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)

        self.assertEqual(val_est_opt["execution_options"]["shots"], 1024)

    def test_validate_estimator_options_terra(self):
        est_prim_str = "terra"
        est_opt = {"backend_options": None}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"run_options": None}
        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)

        self.assertEqual(val_est_opt["run_options"], {})

        est_opt = {"run_options": 1}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)
        
    def test_validate_estimator_options_ion_trap(self):
        est_prim_str = "ion_trap"
        backend_access_path = os.path.join(os.getcwd(), "tests/", "ion_trap_access_dummy.txt")
        est_opt = {"backend_access_path": backend_access_path, "run_options": None, "transpilation_options": None, "abelian_grouping": False, "bound_pass_manager": None, "skip_transpilation": True}
        val_est_opt = self.estimator_cal._validate_estimator_options(est_opt, est_prim_str)

        self.assertEqual(val_est_opt["run_options"], {"shots": 100})
        self.assertIsInstance(val_est_opt["transpilation_options"], dict)

        est_opt = {"backend_access_path": backend_access_path, "run_options": None, "transpilation_options": None, "abelian_grouping": "True", "bound_pass_manager": None, "skip_transpilation": True}
        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"backend_access_path": backend_access_path, "run_options": None, "transpilation_options": None, "abelian_grouping": False, "bound_pass_manager": None, "skip_transpilation": "False"}
        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)

        est_opt = {"backend_access_path": 0, "run_options": None, "transpilation_options": None, "abelian_grouping": False, "bound_pass_manager": None, "skip_transpilation": True}
        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)


    def test_validate_estimator_options_est_prim_str(self):
        est_prim_str = "unkown_estimator"
        est_opt = {}

        self.assertRaises(ValueError, self.estimator_cal._validate_estimator_options, est_opt, est_prim_str)


class TestVQEEstimator(unittest.TestCase):
    def setUp(self):
        self.est_opt = {"abelian_grouping": False, "transpilation_options": {"optimization_level": 3}, "backend_options": {"method": "automatic", "shots": 4000}, "run_options": {"shots": 1024}, "approximation": False, "skip_transpilation": False}
        self.estimator_cal = VQEE.EstimatorCalibration(self.est_opt, "None", "None", "None", "aer", "aer_automatic")
        self.vqe_estimator = VQEE.VQEEstimator(self.estimator_cal)


    def test_init(self):
        est_opt = {"optimization_level": 0, "resilience_level": 0, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {}, "environment_options": {}, "simulator_options": {}}
        
        est_cal = VQEE.EstimatorCalibration(est_opt, "None", "None", "None", "ibm_runtime", "ibm_runtime_qasm")

        self.assertRaises(ValueError, VQEE.VQEEstimator, est_cal)

    def test_repr(self):
        self.assertEqual(repr(self.vqe_estimator), "VQEEstimator(estimator_parameters={}, session={})".format(self.vqe_estimator.parameters, self.vqe_estimator.session))

    def test_to_dic(self):
        self.assertEqual(self.vqe_estimator.to_dict(), {"parameters": self.vqe_estimator.parameters.to_dict(), "session": self.vqe_estimator.session, "estimator": self.vqe_estimator.estimator})

    def test_parameters_setter(self):
        new_est_opt = {"abelian_grouping": True, "transpilation_options": {"optimization_level": 0}, "backend_options": {"method": "automatic", "shots": 4000}, "run_options": {}, "approximation": False, "skip_transpilation": False}
        est_cal = VQEE.EstimatorCalibration(new_est_opt, "None", "None", "None", "aer", "aer_automatic")
        self.vqe_estimator.parameters = est_cal

        print("Aer Estimator object:")
        print(self.vqe_estimator.estimator)
        self.assertEqual(self.vqe_estimator.parameters.estimator_options, new_est_opt)

    # IBM runtime does not support qiskit version < 1.0.0 anymore
    # def test_get_estimator_ibm_runtime(self):
    # 
    #     est_opt = {"optimization_level": 1, "resilience_level": 0, "max_execution_time": None, "transpilation_options": {}, "resilience_options": {}, "execution_options": {"shots": 4000}, "environment_options": {}, "simulator_options": {}}
    #     backend_str = "ibmq_qasm_simulator"
    #     est_cal = VQEE.EstimatorCalibration(est_opt, "None", "None", "None", "ibm_runtime", backend_str)
    #     service = QiskitRuntimeService()
    #     
    #     
    #     with Session(service=service, backend=backend_str) as session:
    #         vqe_est = VQEE.VQEEstimator(est_cal, session=session)
    # 
    #         print("ibm runtime Estimator object:")
    #         print(vqe_est.estimator)

    def test_get_estimator_terra(self):
        est_opt = {"run_options": {"shots": None}}
        est_cal = VQEE.EstimatorCalibration(est_opt, "None", "None", "None", "terra", "statevector")
        vqe_est = VQEE.VQEEstimator(est_cal)

        print("terra Estimator object:")
        print(vqe_est.estimator)
