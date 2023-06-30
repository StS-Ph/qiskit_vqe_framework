import unittest
import src
import src.VQEOptimizer as VQEO
import src.TerminationChecker as tc
import qiskit.algorithms.optimizers as optimizers
import copy

class TestVQEOptimizerCalibration(unittest.TestCase):
    def setUp(self):
        self.checker = tc.RelativeEnergyChecker(100, 20, 0.01)
        self.opt_cal = VQEO.OptimizerCalibration("SPSA", 100, "fin_diff", param_map_init = [0.0, 0.0, 0.0], termination_checker = self.checker)

    def test_param_map_init_setter(self):
        opt_cal  = copy.copy(self.opt_cal)

        self.assertEqual(opt_cal.use_custom_param_init, True)

        opt_cal.param_map_init = None

        self.assertEqual(opt_cal.use_custom_param_init, False)

    def test_repr(self):

        self.assertEqual(repr(self.opt_cal), "OptimizerCalibration(name_str=SPSA, maxiter=100, grad_meth=fin_diff, param_map_init=[0.0, 0.0, 0.0], termination_checker={})".format(self.checker))

    def test_to_dict(self):
        self.assertEqual(self.opt_cal.to_dict(), {"name": "OptimizerCalibration", "optimizer_name": "SPSA", "maxiter": 100, "grad_meth": "fin_diff", "termination_checker": self.checker, "param_map_init": [0.0, 0.0, 0.0], "use_custom_param_init": True})

    def test_get_filevector(self):
        header, data = self.opt_cal.get_filevector()

        self.assertEqual(header, ["optimizer", "opt_max_iter", "grad_method", "use_custom_param_init", "termination_checker"])

        self.assertEqual(data, ["SPSA", 100, "fin_diff", True, self.checker.name])

class TestVQEOptimizer(unittest.TestCase):
    def setUp(self):
        self.checker = tc.RelativeEnergyChecker(100, 20, 0.01)
        self.opt_cal = VQEO.OptimizerCalibration("SPSA", 100, "fin_diff", param_map_init = [0.0, 0.0, 0.0], termination_checker = self.checker)
        self.vqe_optimizer = VQEO.VQEOptimizer(self.opt_cal)

    def test_repr(self):
        self.assertEqual(repr(self.vqe_optimizer), "VQEOptimizer(optimizer_parameters={})".format(self.vqe_optimizer.parameters))

    def test_to_dict(self):
        self.assertEqual(self.vqe_optimizer.to_dict(), {"parameters": self.opt_cal.to_dict(), "optimizer": self.vqe_optimizer.optimizer})

    def test_update_parameters(self):
        opt_cal = copy.copy(self.opt_cal)
        opt_cal.maxiter = 500

        vqe_optimizer = copy.copy(self.vqe_optimizer)
        vqe_optimizer.update_parameters(opt_cal)


        self.assertEqual(vqe_optimizer.parameters.maxiter, 500)
        self.assertEqual(vqe_optimizer.optimizer.maxiter, 500)

    def test_get_optimizer(self):
        self.assertEqual(self.opt_cal.maxiter, self.vqe_optimizer.optimizer.maxiter)
        self.assertEqual(self.opt_cal.termination_checker, self.vqe_optimizer.optimizer.termination_checker)
        self.assertEqual(self.opt_cal.optimizer_name, self.vqe_optimizer.optimizer.__class__.__name__)

        opt_cal = copy.copy(self.opt_cal)
        opt_cal.optimizer_name = "LBFGS"

        self.assertRaises(ValueError, VQEO.VQEOptimizer, opt_cal)

    def test_get_gradient(self):

        self.assertRaises(NotImplementedError, self.vqe_optimizer.get_gradient, None)
