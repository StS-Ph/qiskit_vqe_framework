import unittest
import qiskit_vqe_framework
import qiskit_vqe_framework.TerminationChecker as TC
import copy
import numpy as np

class TestRelativeEnergyChecker(unittest.TestCase):
    def setUp(self):
        self.checker = TC.RelativeEnergyChecker(100, 20, 0.001)

    def test_calc_relative_change1(self):
        x = np.linspace(50, 100, num = 100)
        y = np.sqrt(1/x) + 3

        self.checker.values = y

        val = np.sqrt(1/100.5) + 3

        rel_change = []
        for old, new in zip(y[:], y[1:]):
            delta = np.abs((new-old)/new)
            rel_change.append(delta)

        self.assertEqual(self.checker._calc_relative_change(), rel_change[-self.checker.considered_values_length:])

        self.assertEqual(len(self.checker._calc_relative_change()), self.checker.considered_values_length)

    def test_calc_relative_change2(self):
        x = np.linspace(50, 100, num = self.checker.considered_values_length)
        y = np.sqrt(1/x) + 3

        self.checker.values = y

        val = np.sqrt(1/100.5) + 3

        rel_change = []
        for old, new in zip(y[:], y[1:]):
            delta = np.abs((new-old)/new)
            rel_change.append(delta)

        self.assertEqual(self.checker._calc_relative_change(), rel_change[-self.checker.considered_values_length:])

        self.assertNotEqual(len(self.checker._calc_relative_change()), self.checker.considered_values_length)

    def test_check_termination1(self):
        x = np.linspace(50, 100, num = 100)
        y = np.sqrt(1/x) + 3

        self.checker.values = y

        self.assertEqual(self.checker._check_termination(101, [0.0, 0.0], 100.5, 0.5, True), True)

    def test_check_termination2(self):
        x = np.linspace(50, 100, num = 100)
        y = np.sqrt(x)

        self.checker.values = y

        self.assertEqual(self.checker._check_termination(101, [0.0, 0.0], 100.5, 0.5, True), False)

    def test_call(self):
        x = np.linspace(50, 100, num = 99)
        y = np.sqrt(1/x) + 3

        val = np.sqrt(1/100.5) + 3

        self.checker.values = list(y)

        self.assertEqual(self.checker(101, [0.0, 0.0], val, 0.5, True), True)

        
        
class TestLinearFitChecker(unittest.TestCase):
    def setUp(self):
        self.checker = TC.LinearFitChecker(100, 0.001)

    def test_check_termination1(self):
        x = np.linspace(50, 100, num = 100)
        y = np.sqrt(1/x) + 3

        self.checker.values = y

        self.assertEqual(self.checker._check_termination(101, [0.0, 0.0], 100.5, 0.5, True), True)

    def test_check_termination2(self):
        x = np.linspace(50, 100, num = 100)
        y = 0.5 * x

        self.checker.values = y

        self.assertEqual(self.checker._check_termination(101, [0.0, 0.0], 100.5, 0.5, True), False)

    def test_call(self):
        x = np.linspace(50, 100, num = 99)
        y = np.sqrt(1/x) + 3

        val = np.sqrt(1/100.5) + 3

        self.checker.values = list(y)

        self.assertEqual(self.checker(101, [0.0, 0.0], val, 0.5, True), True)
