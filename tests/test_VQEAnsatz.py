import unittest
import src
import src.VQEAnsatz as VQEA
import copy


class TestVQEAnsatzCalibration(unittest.TestCase):
    def setUp(self):
        self.ansatz_cal = VQEA.AnsatzCalibration(2, 1, "ESU2", [1.0,0.0,0.0,0.0])

    def test_repr(self):
        self.assertEqual(repr(self.ansatz_cal), "AnsatzCalibration(num_qubits=2, num_layers=1, ansatz_str=ESU2, psi_start=[1.0, 0.0, 0.0, 0.0])")

    def test_to_dict(self):
        self.assertEqual(self.ansatz_cal.to_dict(), {"name": "AnsatzCalibration", "num_qubits": 2, "num_layers": 1, "ansatz_str": "ESU2", "psi_start": [1.0,0.0,0.0,0.0], "use_custom_state_init": True})

    def test_get_filevector(self):
        header, data = self.ansatz_cal.get_filevector()

        self.assertEqual(header, ["num_qubits", "num_layers", "ansatz", "use_custom_state_init"])
        self.assertEqual(data, [2, 1, "ESU2", True])

    def test_set_psi_start(self):
        self.ansatz_cal.psi_start = None
        self.assertEqual(self.ansatz_cal.use_custom_state_init, False)

class TestVQEAnsatzESU2(unittest.TestCase):
    def setUp(self):
        self.esu2_ansatz = VQEA.ESU2(2, reps = 1, su2_gates=["rx", "ry"], initial_state = [1.0,0.0,0.0,0.0])

    def test_repr(self):
        self.assertEqual(repr(self.esu2_ansatz), "VQEAnsatz(parameters={})".format(self.esu2_ansatz.parameters))

    def test_to_dict(self):
        self.assertEqual(self.esu2_ansatz.to_dict(), {"parameters": self.esu2_ansatz.parameters.to_dict(), "circuit": self.esu2_ansatz.circuit})

    def test_update_parameters(self):
        esu2_cal_new = copy.copy(self.esu2_ansatz.parameters)
        esu2_cal_new.num_layers = 2
        esu2_cal_new.insert_barriers = True

        esu2_ansatz = copy.copy(self.esu2_ansatz)
        print(esu2_ansatz)
        print(esu2_ansatz.circuit)

        esu2_ansatz.update_parameters(esu2_cal_new)
        print(esu2_ansatz)
        print(esu2_ansatz.circuit)

        self.assertNotEqual(self.esu2_ansatz.to_dict(), esu2_ansatz.to_dict())
        self.assertNotEqual(self.esu2_ansatz.circuit, esu2_ansatz.circuit)
