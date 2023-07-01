import unittest
import qiskit_vqe_framework
import qiskit_vqe_framework.VQETargetModel as VQETM
import copy
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.opflow import PauliSumOp

class TestVQEModelCalibration(unittest.TestCase):
    def setUp(self):
        self.model_cal = VQETM.ModelCalibration("transverse_field_Ising_model", num_spins=4, J=1.0, g=-0.5)

    def test_repr(self):
        self.assertEqual(repr(self.model_cal), "ModelCalibration(model_name=transverse_field_Ising_model, num_spins=4, J=1.0, g=-0.5)")

    def test_to_dict(self):
        self.assertEqual(self.model_cal.to_dict(), {"name": "ModelCalibration", "model_name": "transverse_field_Ising_model", "num_spins": 4, "J": 1.0, "g": -0.5})

    def test_get_filevector(self):
        header, data = self.model_cal.get_filevector()

        self.assertEqual(header, ["model_name", "num_spins", "J", "g"])
        self.assertEqual(data, ["transverse_field_Ising_model", 4, 1.0, -0.5])
        
class TestVQETargetModelTFIM(unittest.TestCase):
    def setUp(self):
        self.tfim = VQETM.TransverseFieldIsingModel(4, J=1.0, g=-0.5)
        
    def test_repr(self):
        self.assertEqual(repr(self.tfim), "VQETargetModel(model_parameters={})".format(self.tfim.parameters))

    def test_to_dict(self):
        self.assertEqual(self.tfim.to_dict(), {"parameters": self.tfim.parameters.to_dict(), "hamiltonian": self.tfim.hamiltonian, "aux_ops": self.tfim.aux_ops})

    def test_update_parameters(self):
        tfim_cal_new = copy.copy(self.tfim.parameters)
        tfim_cal_new.J = 5.0
        tfim_cal_new.g = -2.5

        tfim = copy.copy(self.tfim)
        print(tfim)
        print(tfim.hamiltonian)
        print(tfim.aux_ops)

        self.assertEqual(self.tfim.to_dict(), tfim.to_dict())
        self.assertEqual(self.tfim.parameters.J, tfim.parameters.J)
        self.assertEqual(self.tfim.parameters.g, tfim.parameters.g)
        self.assertEqual(self.tfim.hamiltonian, tfim.hamiltonian)
        self.assertEqual(self.tfim.aux_ops, tfim.aux_ops)
        
        tfim.update_parameters(tfim_cal_new)
        print(tfim)
        print(tfim.hamiltonian)
        print(tfim.aux_ops)

        self.assertNotEqual(self.tfim.to_dict(), tfim.to_dict())
        self.assertNotEqual(self.tfim.parameters.J, tfim.parameters.J)
        self.assertNotEqual(self.tfim.parameters.g, tfim.parameters.g)
        self.assertNotEqual(self.tfim.hamiltonian, tfim.hamiltonian)
        self.assertEqual(self.tfim.aux_ops, tfim.aux_ops)

    def test_get_hamiltonian(self):
        # transverse field Ising Hamiltonian
        J = 1.0
        H = PauliSumOp(SparsePauliOp("ZZII",J))
        H = H.add(PauliSumOp(SparsePauliOp("IZZI",J)))
        H = H.add(PauliSumOp(SparsePauliOp("IIZZ",J)))
        g = -0.5
        H = H.add(PauliSumOp(SparsePauliOp("XIII",g)))
        H = H.add(PauliSumOp(SparsePauliOp("IXII",g)))
        H = H.add(PauliSumOp(SparsePauliOp("IIXI",g)))
        H = H.add(PauliSumOp(SparsePauliOp("IIIX",g)))

        self.assertEqual(self.tfim.hamiltonian, H)

    def test_get_aux_ops(self):
        L = 4
        qtot = PauliSumOp(SparsePauliOp("Z"+("I" * (L-1)), 1/2))
        for l in range(1,L):
            # generate Pauli string
            IL = "I" * l
            IR = "I" * (L-1-l)
            qtot =qtot.add(PauliSumOp(SparsePauliOp(IL+"Z"+IR,1/2)))
        
        aux_ops = {'qtot': qtot}

        self.assertEqual(self.tfim.aux_ops, aux_ops)
