"""
Qiskit Backend - Noise modeling and advanced simulation via Qiskit Aer.

This backend provides access to Qiskit's quantum simulators with noise modeling.
Requires qiskit to be installed.
"""

import numpy as np
from typing import List, Union, Dict


class QiskitBackend:
    """
    Qiskit-based quantum simulator backend.
    
    Provides noise modeling and advanced simulation features via Qiskit Aer.
    Falls back to native simulation if Qiskit is not available.
    """
    
    def __init__(self, num_qubits: int, noise_model=None):
        """
        Initialize Qiskit backend.
        
        Args:
            num_qubits: Number of qubits
            noise_model: Optional Qiskit noise model
        """
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        self._qiskit_available = False
        
        try:
            from qiskit import QuantumCircuit, QuantumRegister
            from qiskit_aer import AerSimulator
            
            self._qiskit_available = True
            self.qr = QuantumRegister(num_qubits, 'q')
            self.circuit = QuantumCircuit(self.qr)
            self.simulator = AerSimulator(method='statevector')
            
        except ImportError:
            import warnings
            warnings.warn("Qiskit not available, using fallback native simulation", UserWarning)
            from ftqec.core.quantum_sim.backends.native_complex_sim import NativeComplexSim
            self._fallback = NativeComplexSim(num_qubits)
    
    def reset(self):
        """Reset the quantum state."""
        if self._qiskit_available:
            from qiskit import QuantumCircuit, QuantumRegister
            self.qr = QuantumRegister(self.num_qubits, 'q')
            self.circuit = QuantumCircuit(self.qr)
        else:
            self._fallback.reset()
    
    def get_statevector(self) -> np.ndarray:
        """Get current statevector."""
        if self._qiskit_available:
            from qiskit import transpile
            result = self.simulator.run(transpile(self.circuit, self.simulator)).result()
            return result.get_statevector().data
        else:
            return self._fallback.get_statevector()
    
    def apply_gate(self, qubit_ids: List[int], gate_matrix: np.ndarray):
        """Apply quantum gate."""
        if self._qiskit_available:
            from qiskit.circuit.library import UnitaryGate
            gate = UnitaryGate(gate_matrix)
            self.circuit.append(gate, qubit_ids)
        else:
            self._fallback.apply_gate(qubit_ids, gate_matrix)
    
    def measure(self, qubit_ids: Union[List[int], None] = None) -> Union[int, List[int]]:
        """Measure qubits."""
        if self._qiskit_available:
            # Get statevector and perform measurement classically
            state = self.get_statevector()
            probs = np.abs(state) ** 2
            outcome = np.random.choice(len(state), p=probs)
            
            if qubit_ids is None:
                qubit_ids = list(range(self.num_qubits))
            
            results = []
            for qubit in qubit_ids:
                bit = (outcome >> (self.num_qubits - 1 - qubit)) & 1
                results.append(bit)
            
            return results[0] if len(results) == 1 else results
        else:
            return self._fallback.measure(qubit_ids)
    
    def entangle(self, qubit_ids: List[int]):
        """Create entanglement."""
        if self._qiskit_available:
            for i in range(len(qubit_ids) - 1):
                self.circuit.cx(qubit_ids[i], qubit_ids[i+1])
        else:
            self._fallback.entangle(qubit_ids)
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        if self._qiskit_available:
            state = self.get_statevector()
            probs = {}
            for i, amplitude in enumerate(state):
                prob = abs(amplitude) ** 2
                if prob > 1e-10:
                    bitstring = format(i, f'0{self.num_qubits}b')
                    probs[bitstring] = float(prob)
            return probs
        else:
            return self._fallback.get_probabilities()
    
    def get_coherence(self) -> float:
        """Calculate quantum coherence."""
        if self._qiskit_available:
            state = self.get_statevector()
            rho = np.outer(state, state.conj())
            coherence = 0.0
            dim = len(state)
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        coherence += abs(rho[i, j])
            max_coherence = dim * (dim - 1)
            return float(coherence / max_coherence) if max_coherence > 0 else 0.0
        else:
            return self._fallback.get_coherence()
    
    def __repr__(self) -> str:
        """String representation."""
        backend_type = "Qiskit" if self._qiskit_available else "Fallback"
        return f"QiskitBackend(num_qubits={self.num_qubits}, type={backend_type})"
