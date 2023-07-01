A unpublished project to provide a data framework for [running vqe simulations in qiskit] [vqe-qiskit].
It provides a calibration class for each major part in a qiskit vqe simualtion (target model, circuit ansatz, optimizer and estimator). Which can be used to easily save all calibration data in seperate files as dictionaries or combined as a filevector in one file.
Each vqe part has additionally a class which contains the calibration and the corresponding qiskit object. The qiskit objects are generated via methods of this class taking the calibration class as an input. This way calibration and corresponding qiskit object are correctly associated with each other.

[vqe-qiskit]: https://qiskit.org/ecosystem/ibm-runtime/tutorials/vqe_with_estimator.html

